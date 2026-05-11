"""
페르소나 데이터 모델 (Persona Models)

C#으로 비유하면 Models/ 폴더의 DTO(Data Transfer Object) 클래스들.
Pydantic의 BaseModel = C#의 record class + DataAnnotations 유효성 검사 합친 것.
"""

# Enum = C#의 enum과 동일
from enum import Enum
# Optional = C#의 nullable type (string? 같은 것)
from typing import Optional
# BaseModel = C#의 class + [Required], [Range] 같은 DataAnnotations 합친 것
# Field = C#의 [Required], [Range(1, 200)] 같은 속성 데코레이터
from pydantic import BaseModel, Field


# C#의 public enum Gender { Male, Female } 와 동일
# str을 상속받아서 JSON 직렬화 시 문자열로 변환됨
class Gender(str, Enum):
    MALE = "male"       # 남성
    FEMALE = "female"   # 여성


# C#의 public enum AgeGroup { ... } 와 동일
# 한국 연령대 구분 (10대~60대 이상)
class AgeGroup(str, Enum):
    TEENS = "10대"
    TWENTIES = "20대"
    THIRTIES = "30대"
    FORTIES = "40대"
    FIFTIES = "50대"
    SIXTIES_PLUS = "60대+"


# C#의 public class PersonaAttributes { ... } 와 동일
# 페르소나의 인구통계학적 속성을 담는 클래스
class PersonaAttributes(BaseModel):
    """페르소나의 핵심 인적 속성 (C#의 DTO 클래스)"""

    age_group: AgeGroup                                # 연령대 (필수)
    gender: Gender                                      # 성별 (필수)
    occupation: str                                     # 직업 (필수)
    education: Optional[str] = None                    # 학력 (선택) - C#의 string? education
    income_level: Optional[str] = None                 # 소득 수준 (선택)
    region: Optional[str] = None                       # 거주 지역 (선택)

    # 행동/성격 특성 리스트
    # Field(default_factory=list) = C#의 List<string> Traits { get; set; } = new();
    traits: list[str] = Field(default_factory=list)     # 성격 특성 (예: ["실용적", "트렌디"])
    interests: list[str] = Field(default_factory=list)  # 관심사 (예: ["요리", "여행"])


# C#의 public class PersonaConfig { ... } - 페르소나 생성 시 조건을 담는 클래스
class PersonaConfig(BaseModel):
    """페르소나 생성 설정 (C#의 Request DTO)"""

    # ge=1, le=200 → C#의 [Range(1, 200)] 과 동일
    panel_count: int = Field(ge=1, le=200, default=10)  # 생성할 패널(응답자) 수

    # 타겟 분포 설정 (선택) - 지정 안 하면 실제 한국 인구 통계 비율 사용
    age_groups: list[AgeGroup] = Field(default_factory=list)            # 포함할 연령대
    gender_ratio: dict[str, float] = Field(                            # 성비 (기본 5:5)
        default_factory=lambda: {"male": 0.5, "female": 0.5}
    )
    occupations: list[str] = Field(default_factory=list)                # 포함할 직업군
    traits: list[str] = Field(default_factory=list)                     # 포함할 성격 특성

    # True면 KOSIS 실제 통계 데이터 기반으로 분포 생성
    use_statistical_distribution: bool = True


# C#의 public class Persona { ... } - 생성된 페르소나 한 명의 전체 데이터
class Persona(BaseModel):
    """생성된 페르소나 (C#의 Entity 클래스)"""

    id: str                                     # 고유 ID (C#의 Guid.NewGuid().ToString())
    name: str                                   # 이름 (예: "김민준")
    attributes: PersonaAttributes               # 인적 속성 (위에서 정의한 클래스)

    weight: float = 1.0                         # 캘리브레이션 가중치 (통계 보정용)
    backstory: Optional[str] = None             # AI가 생성한 배경 스토리 (선택)

    def get_prompt_context(self) -> str:
        """
        AI 프롬프트용 컨텍스트 문자열 생성
        C#의 ToString() 오버라이드와 비슷 - AI에게 "너는 이런 사람이야"라고 알려주는 텍스트
        """
        return f"""
당신은 다음과 같은 사람입니다:
- 이름: {self.name}
- 연령대: {self.attributes.age_group.value}
- 성별: {"남성" if self.attributes.gender == Gender.MALE else "여성"}
- 직업: {self.attributes.occupation}
- 특성: {", ".join(self.attributes.traits) if self.attributes.traits else "없음"}

{self.backstory or ""}
""".strip()
