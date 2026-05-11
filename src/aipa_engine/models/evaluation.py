"""
평가 스키마 모델 (Evaluation Schema Models)

AIPA-Eval 파인튜닝 모델의 입력/출력 데이터 구조를 정의.
C#으로 비유하면 ML.NET 학습 데이터의 Input/Output 클래스 역할.

핵심 흐름:
  자극물(제품/광고 등) + 페르소나 정보 → [AIPA-Eval 모델] → 축별 점수 + 근거
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# 평가할 수 있는 자극물 유형 (C#의 enum StimulusType 와 동일)
# 각 유형마다 기본 평가 축이 다름 (아래 DEFAULT_AXES에서 정의)
class StimulusType(str, Enum):
    """자극물 유형 (평가 대상의 카테고리)"""
    FOOD_PRODUCT = "식품"
    COSMETICS = "화장품"
    APP_SERVICE = "앱/서비스"
    ADVERTISEMENT = "광고"
    INSURANCE = "보험/금융"
    EDUCATION = "교육"
    FASHION = "패션"
    CONTENT = "콘텐츠"
    POLICY = "정책"
    EVENT = "이벤트/프로모션"
    REAL_ESTATE = "부동산"
    AUTOMOBILE = "자동차"
    ELECTRONICS = "가전/전자"
    BUSINESS_PLAN = "사업계획서"
    SURVEY = "설문지"
    GENERAL = "기타"


# 카테고리별 기본 평가 축 (Dictionary<StimulusType, string[]> 같은 것)
# 유저가 커스텀 축을 지정하지 않으면 이 기본값 사용
# 예: 식품이면 ["호감도", "구매의향", "가격적절성", "추천의향", "차별성"]
DEFAULT_AXES = {
    StimulusType.FOOD_PRODUCT: ["호감도", "구매의향", "가격적절성", "추천의향", "차별성"],
    StimulusType.COSMETICS: ["호감도", "구매의향", "가격적절성", "성분신뢰도", "재구매의향"],
    StimulusType.APP_SERVICE: ["사용의향", "편의성", "필요성", "디자인호감도", "추천의향"],
    StimulusType.ADVERTISEMENT: ["주목도", "메시지전달력", "브랜드연상", "호감도", "클릭의향"],
    StimulusType.INSURANCE: ["안전성", "보장범위", "가격적절성", "신뢰도", "가입의향"],
    StimulusType.EDUCATION: ["학습효과", "흥미도", "난이도적절성", "가격적절성", "추천의향"],
    StimulusType.FASHION: ["디자인호감도", "구매의향", "가격적절성", "트렌드부합", "착용의향"],
    StimulusType.CONTENT: ["흥미도", "몰입도", "공감도", "공유의향", "재소비의향"],
    StimulusType.POLICY: ["필요성", "실효성", "공정성", "이해도", "지지도"],
    StimulusType.EVENT: ["참여의향", "매력도", "혜택적절성", "공유의향", "재참여의향"],
    StimulusType.REAL_ESTATE: ["입지매력도", "가격적절성", "투자가치", "거주의향", "추천의향"],
    StimulusType.AUTOMOBILE: ["디자인호감도", "성능기대", "가격적절성", "구매의향", "브랜드신뢰"],
    StimulusType.ELECTRONICS: ["기능매력도", "가격적절성", "구매의향", "브랜드신뢰", "추천의향"],
    StimulusType.BUSINESS_PLAN: ["시장성", "실현가능성", "차별성", "수익성", "리스크"],
    StimulusType.SURVEY: ["응답용이성", "질문명확성", "주제관심도", "완료의향", "소요시간적절성"],
    StimulusType.GENERAL: ["호감도", "관심도", "필요성", "추천의향", "차별성"],
}


# 평가 축 하나의 결과 (예: 호감도 72점, 이유: "...")
class EvaluationAxis(BaseModel):
    """평가 축 하나의 결과 (C#의 ScoreItem 같은 것)"""
    name: str                                                           # 축 이름 (예: "호감도")
    score: int = Field(ge=0, le=100, description="0-100 점수")         # 0~100점
    reasoning: str = Field(description="이 점수를 준 이유 (페르소나 관점에서)")  # 근거 설명


# 평가 요청 = 모델에 넣는 입력 데이터 (C#의 PredictionInput 같은 것)
class EvaluationRequest(BaseModel):
    """평가 요청 (모델 입력) - 자극물 + 페르소나 정보"""

    # --- 자극물 (평가 대상) ---
    stimulus: str = Field(description="평가할 자극물 설명/내용")    # 예: "새로운 단백질 쉐이크 제품..."
    stimulus_type: StimulusType = StimulusType.GENERAL              # 유형

    # --- 페르소나 정보 (어떤 사람이 평가하는지) ---
    persona_age_group: str = Field(description="연령대 (예: 20대)")
    persona_gender: str = Field(description="성별 (male/female)")
    persona_occupation: str = Field(default="", description="직업")
    persona_income: str = Field(default="", description="소득 수준")
    persona_traits: list[str] = Field(default_factory=list, description="특성 키워드")
    persona_backstory: str = Field(default="", description="페르소나 배경 설명")

    # --- 평가 축 (유저가 커스텀 지정 가능) ---
    axes: list[str] = Field(
        default_factory=list,
        description="평가할 축 이름 리스트. 비어있으면 stimulus_type에 따라 자동 설정"
    )

    def get_axes(self) -> list[str]:
        """평가 축 반환 - 유저가 지정한 게 있으면 그걸 사용, 없으면 기본값"""
        if self.axes:
            return self.axes
        return DEFAULT_AXES.get(self.stimulus_type, DEFAULT_AXES[StimulusType.GENERAL])


# 평가 결과 = 모델의 출력 데이터 (C#의 PredictionOutput 같은 것)
class EvaluationResponse(BaseModel):
    """평가 결과 (모델 출력) - 축별 점수 + 자유 응답"""

    # 각 평가 축별 점수와 근거 (예: [호감도 72점, 구매의향 45점, ...])
    evaluations: list[EvaluationAxis]

    # 페르소나의 자연스러운 한 줄 반응
    # 예: "이 제품 괜찮아 보이긴 한데 가격이 좀 비싼 것 같아요"
    open_response: str = Field(description="페르소나의 자연스러운 한 줄 반응")

    # 이 평가의 신뢰도 (0~1, 캘리브레이션 후 조정됨)
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.7,
        description="이 평가의 신뢰도 (캘리브레이션 후 조정됨)"
    )


# 파인튜닝 학습 데이터 한 건 (C#의 TrainingData 클래스 같은 것)
# input(질문) + output(정답) 쌍으로 구성 → 이걸 3000개 만들어서 모델 학습
class TrainingExample(BaseModel):
    """파인튜닝용 학습 데이터 한 건 (입력-출력 쌍)"""
    input: EvaluationRequest        # 모델에 넣을 입력
    output: EvaluationResponse      # 모델이 내놓아야 할 정답

    def to_prompt_completion(self) -> dict[str, str]:
        """
        SFT(Supervised Fine-Tuning) 학습용 prompt-completion 쌍으로 변환
        C#의 ToTrainingFormat() 같은 변환 메서드

        반환 예시:
        {
            "prompt": "당신은 20대 남성 회사원입니다... 이 제품을 평가해주세요...",
            "completion": '{"evaluations": [...], "open_response": "..."}'
        }
        """
        axes = self.input.get_axes()
        # 페르소나 설명 문자열 조립
        persona_desc = (
            f"{self.input.persona_age_group} {self.input.persona_gender} "
            f"{self.input.persona_occupation}"
        ).strip()
        if self.input.persona_traits:
            persona_desc += f" ({', '.join(self.input.persona_traits)})"

        # 프롬프트 (모델에게 주는 질문)
        prompt = f"""당신은 다음 페르소나입니다:
{persona_desc}
{self.input.persona_backstory}

다음 자극물을 평가해주세요:
[자극물 유형: {self.input.stimulus_type.value}]
{self.input.stimulus}

평가 축: {', '.join(axes)}

각 축에 대해 0-100 점수와 그 이유를 JSON으로 답변하세요."""

        # 정답 (모델이 이렇게 답해야 함) - JSON 문자열로 변환
        completion = self.output.model_dump_json(ensure_ascii=False)

        return {"prompt": prompt, "completion": completion}
