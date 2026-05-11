"""
페르소나 API 엔드포인트 (Persona API Endpoints)

C#으로 비유하면 PersonaController.cs 역할.
FastAPI의 router = C#의 [ApiController] + [Route("api/v1/personas")]
"""

# APIRouter = C#의 Controller 클래스
# HTTPException = C#의 throw new BadRequestException() 같은 것
from fastapi import APIRouter, HTTPException
# BaseModel = C#의 Request DTO 클래스
# Field = C#의 [Range], [Required] 같은 유효성 검사 속성
from pydantic import BaseModel, Field

# 우리가 만든 모델 클래스들 (Models/ 폴더에서 가져옴)
from ..models import PersonaConfig, Persona
from ..models.persona import AgeGroup
# 페르소나 생성 서비스 (C#의 IPersonaService 같은 것)
from ..services.population_generator import PopulationGenerator

# router = C#의 [ApiController, Route("personas")] 같은 것
# 이 라우터에 등록된 모든 엔드포인트는 /api/v1/personas/ 밑에 붙음
router = APIRouter()

# 유효한 연령대 목록을 미리 추출 (["10대", "20대", "30대", ...])
# C#의 Enum.GetValues<AgeGroup>().Select(x => x.ToString()) 같은 것
VALID_AGE_GROUPS = [ag.value for ag in AgeGroup]


# API 요청 바디 DTO (C#의 public class GeneratePersonasRequest { ... })
class GeneratePersonasRequest(BaseModel):
    """페르소나 생성 요청 바디 (C#의 [FromBody] 파라미터)"""

    # ge=1, le=200 → C#의 [Range(1, 200)]
    panel_count: int = Field(default=10, ge=1, le=200)      # 생성할 패널 수 (1~200명)
    age_groups: list[str] = []                                # 원하는 연령대 (예: ["20대", "30대"])
    gender_ratio: dict[str, float] = {"male": 0.5, "female": 0.5}  # 성비 (기본 5:5)
    occupations: list[str] = []                               # 원하는 직업군
    traits: list[str] = []                                    # 원하는 성격 특성
    generate_backstories: bool = False                        # AI 배경스토리 생성 여부


def parse_age_groups(raw: list[str]) -> list[AgeGroup]:
    """
    문자열 연령대를 Enum으로 변환하는 헬퍼 함수
    C#의 Enum.TryParse<AgeGroup>() 같은 역할
    잘못된 값이 들어오면 400 에러 반환

    예: ["20대", "30대"] → [AgeGroup.TWENTIES, AgeGroup.THIRTIES]
    """
    result = []
    for ag in raw:
        try:
            result.append(AgeGroup(ag))  # 문자열 → Enum 변환 시도
        except ValueError:
            # 변환 실패 → 400 Bad Request (C#의 throw new BadRequestException())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid age_group: '{ag}'. Valid values: {VALID_AGE_GROUPS}",
            )
    return result


# POST /api/v1/personas/generate
# C#의 [HttpPost("generate")] public async Task<List<Persona>> Generate([FromBody] request)
@router.post("/generate", response_model=list[Persona])
async def generate_personas(request: GeneratePersonasRequest):
    """페르소나 생성 API - 조건에 맞는 가상 인물 패널을 생성"""

    # 연령대 파싱: 지정된 게 있으면 변환, 없으면 전체 연령대 사용
    age_group_enums = parse_age_groups(request.age_groups) if request.age_groups else list(AgeGroup)

    # 페르소나 생성 설정 객체 조립 (C#의 new PersonaConfig { ... })
    config = PersonaConfig(
        panel_count=request.panel_count,
        age_groups=age_group_enums,
        gender_ratio=request.gender_ratio,
        occupations=request.occupations,
        traits=request.traits,
    )

    # 인구 생성기 인스턴스 생성 후 실행 (C#의 new PopulationGenerator().Generate(config))
    generator = PopulationGenerator()
    personas = await generator.generate(config)  # await = C#의 await 와 완전히 동일

    # 배경스토리 생성이 요청된 경우 (AI가 각 페르소나마다 배경 이야기 작성)
    if request.generate_backstories:
        # 여기서 import 하는 이유: 순환 참조 방지 (C#에서는 이런 문제가 덜하지만 Python에서는 흔함)
        from ..services.llm_service import LLMService

        llm = LLMService()
        for persona in personas:
            persona.backstory = await llm.generate_backstory(persona)

    return personas  # FastAPI가 자동으로 JSON 직렬화 (C#의 Ok(personas) 같은 것)


# GET /api/v1/personas/templates
# C#의 [HttpGet("templates")] public IActionResult GetTemplates()
@router.get("/templates")
async def get_persona_templates():
    """미리 정의된 페르소나 템플릿 목록 반환 (프론트엔드 드롭다운용)"""
    return {
        "templates": [
            {
                "id": "consumer-mix",
                "name": "일반 소비자 믹스",
                "description": "20-40대 남녀, 다양한 배경",
                "config": {
                    "panel_count": 10,
                    "age_groups": ["20대", "30대", "40대"],
                    "gender_ratio": {"male": 0.5, "female": 0.5},
                },
            },
            {
                "id": "mz-workers",
                "name": "MZ세대 직장인",
                "description": "20-30대 회사원 중심",
                "config": {
                    "panel_count": 15,
                    "age_groups": ["20대", "30대"],
                    "occupations": ["회사원", "프리랜서", "스타트업"],
                },
            },
            {
                "id": "seniors",
                "name": "시니어 패널",
                "description": "50-60대 이상",
                "config": {
                    "panel_count": 10,
                    "age_groups": ["50대", "60대+"],
                },
            },
        ]
    }
