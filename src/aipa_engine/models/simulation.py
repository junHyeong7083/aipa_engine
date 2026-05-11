"""
시뮬레이션 세션 모델 (Simulation Session Models)

C#으로 비유하면 시뮬레이션 작업의 상태를 추적하는 Entity 클래스.
웹 요청 → 백그라운드 작업 → 결과 조회 흐름을 관리함.
(C#의 BackgroundService + SignalR 진행률 전송 패턴과 비슷)
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .persona import Persona, PersonaConfig
from .survey import SurveyQuestion, SurveyResponse


# 시뮬레이션 진행 상태 (C#의 enum TaskStatus 같은 것)
class SimulationStatus(str, Enum):
    PENDING = "pending"                          # 대기 중 (아직 시작 안 함)
    GENERATING_PERSONAS = "generating_personas"  # 페르소나 생성 중
    RUNNING_SURVEY = "running_survey"            # 설문 시뮬레이션 실행 중
    CALIBRATING = "calibrating"                  # 통계 보정(캘리브레이션) 중
    COMPLETED = "completed"                      # 완료
    FAILED = "failed"                            # 실패


# C#의 public class SimulationRequest { ... } - API 요청 바디
class SimulationRequest(BaseModel):
    """시뮬레이션 생성 요청 (C#의 [FromBody] CreateSimulationRequest)"""

    persona_config: PersonaConfig           # 페르소나 생성 조건
    questions: list[SurveyQuestion]         # 설문 질문 목록

    enable_calibration: bool = True         # 통계 보정 사용 여부
    generate_explanations: bool = True      # AI 응답 이유 생성 여부


# C#의 public class SimulationProgress { ... } - 진행률 DTO
class SimulationProgress(BaseModel):
    """시뮬레이션 진행 상황 (C#의 IProgress<T> 패턴과 비슷)"""

    status: SimulationStatus                                    # 현재 상태
    progress: float = Field(ge=0.0, le=1.0, default=0.0)      # 진행률 (0.0 ~ 1.0 = 0% ~ 100%)
    message: str = ""                                           # 상태 메시지

    personas_generated: int = 0     # 지금까지 생성된 페르소나 수
    responses_collected: int = 0    # 지금까지 수집된 응답 수


# C#의 public class SimulationSession { ... } - 시뮬레이션 전체 세션 데이터
# 하나의 시뮬레이션 실행에 대한 모든 정보를 담고 있음
class SimulationSession(BaseModel):
    """시뮬레이션 세션 (C#의 Entity + 상태머신 패턴)"""

    id: str                                                     # 세션 고유 ID
    created_at: datetime = Field(default_factory=datetime.utcnow)   # 생성 시각
    updated_at: datetime = Field(default_factory=datetime.utcnow)   # 수정 시각

    status: SimulationStatus = SimulationStatus.PENDING  # 현재 상태
    progress: float = 0.0                                # 진행률 (0~1)

    # 입력 데이터
    config: PersonaConfig                       # 페르소나 생성 조건
    questions: list[SurveyQuestion]             # 설문 질문 목록

    # 출력 데이터 (시뮬레이션 진행하면서 채워짐)
    personas: list[Persona] = Field(default_factory=list)           # 생성된 페르소나들
    responses: list[SurveyResponse] = Field(default_factory=list)   # 수집된 응답들

    # 품질 검증 메트릭 (에러 발생 시 에러 정보도 여기에 저장)
    validation_metrics: Optional[dict] = None


# C#의 public class SimulationResult { ... } - 최종 결과 DTO
class SimulationResult(BaseModel):
    """시뮬레이션 최종 결과 (C#의 Response DTO)"""

    session_id: str     # 어떤 세션의 결과인지

    # 질문별 응답 분포 (예: {"q1": {"매우 그렇다": 0.3, "그렇다": 0.5, ...}})
    # C#의 Dictionary<string, Dictionary<string, double>> 과 동일
    response_distribution: dict[str, dict[str, float]]

    # 질문 ID → 질문 텍스트 매핑
    question_texts: dict[str, str] = {}

    # 개별 응답 상세 (각 페르소나가 각 질문에 뭐라고 답했는지)
    detailed_responses: list[SurveyResponse]

    # 품질 지표 (0.0 ~ 1.0, 1에 가까울수록 좋음)
    distribution_fidelity: float = Field(ge=0.0, le=1.0)    # 분포 충실도 (목표 인구통계와 얼마나 일치하는지)
    consistency_score: float = Field(ge=0.0, le=1.0)        # 일관성 점수

    # 참고용 페르소나 목록
    personas: list[Persona]
