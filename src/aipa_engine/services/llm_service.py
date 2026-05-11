"""
LLM 서비스 (AI 텍스트 생성)

C#으로 비유하면 OpenAI/Claude SDK를 래핑한 Service 클래스.
페르소나의 배경 스토리, 설문 응답 이유 등을 AI가 생성.

사용하는 API: Anthropic Claude (anthropic 패키지)
C#의 HttpClient로 OpenAI API 호출하는 것과 동일한 패턴.
"""

import logging
import time
# Optional = C#의 nullable (string? 같은 것)
from typing import Optional
# anthropic = Anthropic의 Python SDK (C#의 NuGet 패키지 같은 것)
import anthropic

# 대략적인 토큰 추정: 한국어 1글자 ~= 1~2 토큰, 영어 1단어 ~= 1토큰
# Claude 모델의 컨텍스트 제한에 대한 경고 임계값 (문자 수 기준)
_PROMPT_CHAR_WARN_THRESHOLD = 50_000  # ~25k 토큰 추정

# 설정에서 API 키, 모델명 가져오기
from ..config import get_settings
# 페르소나, 성별 Enum 가져오기
from ..models.persona import Persona, Gender
# 설문 질문 모델
from ..models.survey import SurveyQuestion

# C#의 ILogger<LLMService> 같은 것
logger = logging.getLogger(__name__)


class LLMService:
    """
    AI 텍스트 생성 서비스 (C#의 public class LLMService : ILLMService)

    Claude API를 사용해서 자연어 텍스트를 생성.
    API 키가 없으면 Mock(가짜) 데이터를 반환 (개발/테스트용).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        생성자 (C#의 public LLMService(IOptions<Settings> settings))

        api_key, model을 직접 넣거나, 안 넣으면 .env에서 자동으로 가져옴
        """
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key     # API 키
        self.model = model or settings.anthropic_model            # 모델명 (예: claude-sonnet-4-20250514)

        # API 키가 있으면 실제 클라이언트 생성, 없으면 None (Mock 모드)
        if self.api_key:
            # C#의 new HttpClient() + API 키 설정과 비슷
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None  # API 키 없으면 Mock 모드로 동작
            logger.warning("LLMService initialized without API key - all responses will be mock data")

    def _call_api(self, prompt: str, max_tokens: int, max_retries: int = 2) -> str:
        """
        Claude API 호출 + 재시도 로직 (지수 백오프).
        프롬프트 길이 경고 포함.
        """
        if len(prompt) > _PROMPT_CHAR_WARN_THRESHOLD:
            logger.warning(
                f"Prompt is very long ({len(prompt)} chars, ~{len(prompt)//2} tokens est.). "
                "Consider shortening to avoid context limits or high costs."
            )

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text.strip()
            except anthropic.RateLimitError as e:
                last_exc = e
                if attempt < max_retries:
                    delay = (2 ** attempt) + 0.5
                    logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
            except anthropic.APIError as e:
                last_exc = e
                if attempt < max_retries:
                    delay = (2 ** attempt) + 0.5
                    logger.warning(f"API error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Claude API call failed after {max_retries + 1} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}")
                raise
        raise last_exc

    async def generate_backstory(self, persona: Persona) -> str:
        """
        페르소나의 배경 스토리 생성

        예: "김민준님은 30대의 회사원입니다. 트렌디한 성향을 가지고 있으며..."
        API 없으면 Mock 데이터 반환
        """
        # API 클라이언트가 없으면 Mock 데이터 반환
        if not self.client:
            return self._generate_mock_backstory(persona)

        # Claude에게 보낼 프롬프트 작성 (한국어로)
        prompt = f"""다음 인물 프로필을 바탕으로 간결하고 현실적인 배경 스토리를 2-3문장으로 작성해주세요.

인물 프로필:
- 이름: {persona.name}
- 연령대: {persona.attributes.age_group.value}
- 성별: {"남성" if persona.attributes.gender == Gender.MALE else "여성"}
- 직업: {persona.attributes.occupation}
- 특성: {", ".join(persona.attributes.traits) if persona.attributes.traits else "없음"}

배경 스토리:"""

        try:
            return self._call_api(prompt, max_tokens=200)
        except Exception as e:
            logger.error(f"LLM backstory generation failed, using mock: {e}")
            return self._generate_mock_backstory(persona)

    async def generate_response_explanation(
        self,
        persona: Persona,
        question: SurveyQuestion,
        selected_choice: str,
    ) -> str:
        """
        페르소나가 왜 그 답을 선택했는지 AI가 이유를 생성

        예: "제 나이대와 직업을 고려했을 때 '매우 그렇다'가 가장 적합하다고 생각합니다"
        """
        if not self.client:
            return self._generate_mock_explanation(persona, selected_choice)

        # 페르소나 컨텍스트 (AI에게 "너는 이런 사람이야"라고 알려줌)
        context = persona.get_prompt_context()

        prompt = f"""{context}

위 인물이 다음 설문에 답변합니다.

질문: {question.text}
선택지: {", ".join(question.choices)}
선택한 답변: {selected_choice}

왜 이 답변을 선택했는지 이 인물의 관점에서 1-2문장으로 설명해주세요. 1인칭으로 작성하세요."""

        try:
            return self._call_api(prompt, max_tokens=150)
        except Exception as e:
            logger.error(f"LLM response explanation failed, using mock: {e}")
            return self._generate_mock_explanation(persona, selected_choice)

    async def generate_likert_explanation(
        self,
        persona: Persona,
        question: SurveyQuestion,
        scale_value: int,
    ) -> str:
        """리커트 척도(1~5점) 응답에 대한 이유 생성"""
        if not self.client:
            return f"{scale_value}점을 선택했습니다."

        context = persona.get_prompt_context()
        labels = question.scale_labels or {}
        label = labels.get(scale_value, str(scale_value))  # 점수에 해당하는 라벨 (예: 4 → "만족")

        prompt = f"""{context}

위 인물이 다음 설문에 답변합니다.

질문: {question.text}
척도: {question.scale_min} ~ {question.scale_max}
선택한 점수: {scale_value} ({label})

왜 이 점수를 선택했는지 이 인물의 관점에서 1문장으로 설명해주세요."""

        try:
            return self._call_api(prompt, max_tokens=100)
        except Exception as e:
            logger.error(f"LLM likert explanation failed, using fallback: {e}")
            return f"{scale_value}점을 선택했습니다."

    async def generate_open_response(
        self,
        persona: Persona,
        question: SurveyQuestion,
    ) -> str:
        """주관식(개방형) 질문에 대한 응답 생성"""
        if not self.client:
            return f"{persona.name}의 의견입니다."

        context = persona.get_prompt_context()

        prompt = f"""{context}

위 인물이 다음 개방형 질문에 답변합니다.

질문: {question.text}

이 인물의 관점에서 자연스럽게 2-3문장으로 답변해주세요. 1인칭으로 작성하세요."""

        try:
            return self._call_api(prompt, max_tokens=200)
        except Exception as e:
            logger.error(f"LLM open response failed, using fallback: {e}")
            return f"{persona.name}의 의견입니다."

    # === Mock 데이터 생성 (API 없을 때 사용) ===

    def _generate_mock_backstory(self, persona: Persona) -> str:
        """API 없을 때 가짜 배경스토리 반환 (개발/테스트용)"""
        age = persona.attributes.age_group.value
        occupation = persona.attributes.occupation
        traits = ", ".join(persona.attributes.traits[:2]) if persona.attributes.traits else "평범한"

        return f"{persona.name}님은 {age}의 {occupation}입니다. {traits} 성향을 가지고 있으며, 일상에서 다양한 경험을 쌓아가고 있습니다."

    def _generate_mock_explanation(self, persona: Persona, choice: str) -> str:
        """API 없을 때 가짜 응답 이유 반환"""
        return f"제 나이대와 직업을 고려했을 때 '{choice}'가 가장 적합하다고 생각합니다."
