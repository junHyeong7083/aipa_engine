"""
시뮬레이션 API 엔드포인트 (Simulation API Endpoints)

C#으로 비유하면 SimulationController.cs 역할.
시뮬레이션 생성 → 백그라운드 실행 → 상태 조회 → 결과 조회 흐름을 처리.
(C#의 BackgroundService + IHostedService 패턴과 유사)
"""

import logging
# uuid4 = C#의 Guid.NewGuid() 와 동일
from uuid import uuid4
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# 데이터 모델 import
from ..models import (
    SimulationRequest,
    SimulationSession,
    SimulationResult,
    SimulationStatus,
    PersonaConfig,
    SurveyQuestion,
)
# 시뮬레이션 실행 서비스 (C#의 ISimulationService)
from ..services.simulation_service import SimulationService
# Firestore 서비스 (DB 저장)
from ..services.firestore_service import FirestoreService
# 연령대 파싱 함수 (personas.py에서 공유)
from .personas import parse_age_groups

# C#의 ILogger<SimulationController> 같은 것
logger = logging.getLogger(__name__)

router = APIRouter()

# 메모리 내 세션 저장소 (프로덕션에서는 DB로 교체 필요)
# C#의 static Dictionary<string, SimulationSession> sessions = new();
# ⚠️ 서버 재시작하면 데이터 사라짐 → 추후 Firestore로 교체 예정
sessions: dict[str, SimulationSession] = {}


# API 요청 바디 DTO
class CreateSimulationRequest(BaseModel):
    """시뮬레이션 생성 요청 (C#의 [FromBody] CreateSimulationRequest)"""

    panel_count: int = Field(default=10, ge=1, le=200)      # 패널 수
    age_groups: list[str] = []                                # 연령대
    gender_ratio: dict[str, float] = {"male": 0.5, "female": 0.5}  # 성비
    occupations: list[str] = []                               # 직업군
    traits: list[str] = []                                    # 성격 특성
    # min_length=1 → 최소 1개 이상의 질문 필수
    questions: list[dict] = Field(default=[], min_length=1)   # 설문 질문 목록


# POST /api/v1/simulations/
# C#의 [HttpPost] public async Task<SimulationSession> Create([FromBody] request)
@router.post("/", response_model=SimulationSession)
async def create_simulation(
    request: CreateSimulationRequest,
    background_tasks: BackgroundTasks,  # FastAPI의 백그라운드 작업 (C#의 IBackgroundTaskQueue 같은 것)
):
    """시뮬레이션 세션 생성 - 요청을 받으면 바로 세션 ID를 반환하고, 실제 작업은 백그라운드에서 실행"""

    session_id = str(uuid4())  # 고유 세션 ID 생성

    # 연령대 문자열 → Enum 변환
    from ..models.persona import AgeGroup
    age_group_enums = parse_age_groups(request.age_groups) if request.age_groups else list(AgeGroup)

    # 페르소나 설정 객체 조립
    config = PersonaConfig(
        panel_count=request.panel_count,
        age_groups=age_group_enums,
        gender_ratio=request.gender_ratio,
        occupations=request.occupations,
        traits=request.traits,
    )

    # 질문 dict → SurveyQuestion 모델로 변환
    # C#의 request.Questions.Select(q => new SurveyQuestion { ... }).ToList()
    questions = [
        SurveyQuestion(
            id=q.get("id", str(uuid4())),       # ID가 없으면 자동 생성
            text=q.get("text", ""),               # 질문 텍스트
            choices=q.get("choices", []),          # 보기 목록
        )
        for q in request.questions
    ]

    # 세션 객체 생성
    session = SimulationSession(
        id=session_id,
        config=config,
        questions=questions,
        status=SimulationStatus.PENDING,  # 초기 상태: 대기 중
    )

    # 메모리에 세션 저장
    sessions[session_id] = session

    # 백그라운드에서 시뮬레이션 실행 등록
    # C#의 BackgroundService.ExecuteAsync() 같은 것 - 응답 반환 후 별도 쓰레드에서 실행
    background_tasks.add_task(run_simulation, session_id)

    return session  # 세션 ID와 PENDING 상태를 즉시 반환 (클라이언트는 이 ID로 상태 폴링)


# GET /api/v1/simulations/{session_id}
# C#의 [HttpGet("{sessionId}")] public SimulationSession Get(string sessionId)
@router.get("/{session_id}", response_model=SimulationSession)
async def get_simulation(session_id: str):
    """세션 상태 조회 - 클라이언트가 주기적으로 폴링해서 진행률 확인"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


# GET /api/v1/simulations/{session_id}/result
# C#의 [HttpGet("{sessionId}/result")]
@router.get("/{session_id}/result", response_model=SimulationResult)
async def get_simulation_result(session_id: str):
    """완료된 시뮬레이션 결과 조회 - COMPLETED 상태일 때만 조회 가능"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # 아직 완료 안 됐으면 400 에러
    if session.status != SimulationStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Simulation not completed. Current status: {session.status}",
        )

    # 질문별 응답 분포 계산 (아래 함수)
    response_distribution = calculate_distribution(session)

    # 분포 충실도 계산 (목표 인구통계와 실제 결과가 얼마나 일치하는지)
    from ..services.calibrator import Calibrator
    calibrator = Calibrator()
    target_marginals = calibrator._build_target_marginals(session.config)
    fidelity = calibrator.calculate_distribution_fidelity(session.personas, target_marginals)

    # 질문 ID → 텍스트 매핑
    question_texts = {q.id: q.text for q in session.questions}

    return SimulationResult(
        session_id=session_id,
        response_distribution=response_distribution,    # 질문별 응답 비율
        question_texts=question_texts,                   # 질문 ID → 텍스트
        detailed_responses=session.responses,            # 개별 응답 상세
        distribution_fidelity=fidelity,                  # 인구통계 일치도 (0~1)
        consistency_score=fidelity,                      # 일관성 점수 (현재는 동일 값)
        personas=session.personas,                       # 참고용 페르소나 목록
    )


async def run_simulation(session_id: str):
    """
    백그라운드에서 실행되는 시뮬레이션 전체 파이프라인
    C#의 BackgroundService.ExecuteAsync() 같은 역할

    실행 순서:
    1. 페르소나 생성 (통계 기반)
    2. 설문 응답 생성 (AI 기반)
    3. 캘리브레이션 (통계 보정)
    """
    session = sessions.get(session_id)
    if not session:
        return

    try:
        service = SimulationService()

        # === 1단계: 페르소나 생성 ===
        session.status = SimulationStatus.GENERATING_PERSONAS
        session.progress = 0.1  # 10% 진행
        personas = await service.generate_personas(session.config)
        session.personas = personas
        session.progress = 0.4  # 40% 진행

        # === 2단계: 설문 응답 생성 (자체 임베딩 모델 + 0.5B 모델, Claude API 미사용) ===
        session.status = SimulationStatus.RUNNING_SURVEY
        responses = await service.run_survey(personas, session.questions, generate_explanations=True)
        session.responses = responses
        session.progress = 0.8  # 80% 진행

        # === 3단계: 캘리브레이션 (통계 보정) ===
        session.status = SimulationStatus.CALIBRATING
        calibrated_responses = await service.calibrate(responses, session.config, personas)
        session.responses = calibrated_responses
        session.progress = 1.0  # 100% 완료

        session.status = SimulationStatus.COMPLETED

        # Firestore에 완료된 세션 저장
        try:
            fs = FirestoreService()
            if fs.available:
                fs.save_simulation(session_id, session.model_dump(mode="json"))
                logger.info(f"Simulation {session_id} saved to Firestore")
        except Exception as fs_err:
            logger.warning(f"Firestore save failed: {fs_err}")

    except Exception as e:
        # 실패 시 에러 정보 저장
        session.status = SimulationStatus.FAILED
        session.validation_metrics = {"error": str(e)}
        logger.exception(f"Simulation {session_id} failed")


def calculate_distribution(session: SimulationSession) -> dict[str, dict[str, float]]:
    """
    질문별 응답 분포 계산
    C#의 LINQ GroupBy + Aggregate 패턴과 비슷

    예시 결과: {"질문1_id": {"매우 그렇다": 0.3, "그렇다": 0.5, "보통": 0.2}}
    """
    distribution: dict[str, dict[str, float]] = {}

    for question in session.questions:
        # 이 질문에 대한 응답만 필터링 (C#의 .Where(r => r.QuestionId == question.Id))
        question_responses = [r for r in session.responses if r.question_id == question.id]

        if not question_responses:
            distribution[question.id] = {}
            continue

        # 응답별 가중치 합산 (캘리브레이션된 가중치 사용)
        counts: dict[str, float] = {}
        total_weight = 0.0

        for response in question_responses:
            choice = response.selected_choice or ""
            weight = response.weight                    # 캘리브레이션된 가중치
            counts[choice] = counts.get(choice, 0) + weight  # 가중치 누적
            total_weight += weight

        # 비율로 변환 (가중치 합 → 0~1 비율)
        if total_weight > 0:
            distribution[question.id] = {
                choice: count / total_weight for choice, count in counts.items()
            }
        else:
            distribution[question.id] = {}

    return distribution
