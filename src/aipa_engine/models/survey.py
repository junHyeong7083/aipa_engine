"""
설문 데이터 모델 (Survey Models)

C#으로 비유하면 설문 질문(Question)과 응답(Response) DTO 클래스들.
설문 시뮬레이션에서 AI 페르소나가 답변하는 구조를 정의함.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# C#의 public enum QuestionType { ... } 와 동일
# 설문 질문 유형을 정의
class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"        # 단일 선택 (라디오 버튼)
    MULTIPLE_CHOICE = "multiple_choice"    # 복수 선택 (체크박스)
    LIKERT_SCALE = "likert_scale"          # 리커트 척도 (1~5점 같은 점수)
    OPEN_ENDED = "open_ended"              # 주관식 (자유 서술)


# C#의 public class SurveyQuestion { ... } 과 동일
class SurveyQuestion(BaseModel):
    """설문 질문 하나 (C#의 Question DTO)"""

    id: str                                                      # 질문 고유 ID
    text: str                                                    # 질문 텍스트 (예: "이 제품을 구매하시겠습니까?")
    question_type: QuestionType = QuestionType.SINGLE_CHOICE    # 질문 유형 (기본: 단일선택)

    # 객관식 보기 (단일/복수 선택용)
    # 예: ["매우 그렇다", "그렇다", "보통", "아니다", "매우 아니다"]
    choices: list[str] = Field(default_factory=list)

    # 리커트 척도 설정 (1~5점 같은 범위)
    scale_min: int = 1                                          # 최소 점수
    scale_max: int = 5                                          # 최대 점수
    # 점수별 라벨 (예: {1: "매우 불만족", 5: "매우 만족"})
    scale_labels: Optional[dict[int, str]] = None

    # 검증용: 예상 분포 (예: {"매우 그렇다": 0.3, "그렇다": 0.4, ...})
    # 시뮬레이션 결과가 이 분포와 얼마나 비슷한지 비교하는 데 사용
    expected_distribution: Optional[dict[str, float]] = None


# C#의 public class SurveyResponse { ... } 과 동일
class SurveyResponse(BaseModel):
    """페르소나의 설문 응답 하나 (C#의 Response DTO)"""

    persona_id: str     # 응답한 페르소나 ID (C#의 외래키 FK 같은 것)
    question_id: str    # 응답한 질문 ID

    # === 응답 값 (질문 유형에 따라 하나만 채워짐) ===
    selected_choice: Optional[str] = None                       # 단일 선택 응답
    selected_choices: list[str] = Field(default_factory=list)   # 복수 선택 응답
    scale_value: Optional[int] = None                           # 리커트 점수 (예: 4)
    open_response: Optional[str] = None                         # 주관식 응답 텍스트

    # AI가 생성한 응답 이유 설명
    # 예: "제 나이대와 직업을 고려했을 때 '그렇다'가 적합하다고 생각합니다"
    explanation: Optional[str] = None

    # === 통계 메타데이터 ===
    probability: float = 1.0    # 이 응답이 선택될 확률 P(응답 | 페르소나 속성)
    weight: float = 1.0         # 캘리브레이션 가중치 (통계 보정 후 조정됨)

    def get_response_value(self) -> str | int | list[str]:
        """
        질문 유형에 관계없이 응답 값을 반환하는 헬퍼 메서드
        C#의 object GetValue() 같은 역할 - 타입에 따라 적절한 값 반환
        """
        if self.selected_choice:        # 단일 선택이면 그 값 반환
            return self.selected_choice
        if self.selected_choices:       # 복수 선택이면 리스트 반환
            return self.selected_choices
        if self.scale_value is not None:  # 리커트 점수면 숫자 반환
            return self.scale_value
        if self.open_response:          # 주관식이면 텍스트 반환
            return self.open_response
        return ""                       # 아무것도 없으면 빈 문자열
