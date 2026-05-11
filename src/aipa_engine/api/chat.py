"""
채팅 API 엔드포인트 (Chat API Endpoints)

페르소나 기반 AI 채팅 메시지를 처리.
Flutter 앱의 chat_screen에서 호출됨.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.llm_service import LLMService
from ..config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatHistoryItem(BaseModel):
    """대화 히스토리 항목"""
    role: str = Field(..., description="'user' 또는 'assistant'")
    content: str = Field(..., description="메시지 내용")


class ChatMessageRequest(BaseModel):
    """채팅 메시지 요청 바디"""
    message: str = Field(..., min_length=1, description="사용자 메시지")
    persona: dict = Field(default_factory=dict, description="페르소나 정보")
    goal: Optional[str] = Field(None, description="목표 (feedback, evaluate, improve, score, counter)")
    format: Optional[str] = Field(None, description="결과 형식 (short, detailed)")
    history: list[ChatHistoryItem] = Field(default_factory=list, description="이전 대화 히스토리")
    file_path: Optional[str] = Field(None, description="업로드된 파일 경로")
    file_name: Optional[str] = Field(None, description="업로드된 파일명")
    extracted_text: Optional[str] = Field(None, description="파일에서 추출된 텍스트")
    image_base64: Optional[str] = Field(None, description="이미지 base64 데이터")
    image_media_type: Optional[str] = Field(None, description="이미지 MIME 타입")


class ChatMessageResponse(BaseModel):
    """채팅 메시지 응답"""
    response: str
    persona_name: str


# 목표별 한국어 지시문 매핑
GOAL_INSTRUCTIONS = {
    "feedback": "사용자의 내용에 대해 건설적인 피드백을 제공해주세요.",
    "evaluate": "사용자의 내용을 객관적으로 평가해주세요. 강점과 약점을 분석해주세요.",
    "improve": "사용자의 내용에 대한 구체적인 개선안을 제시해주세요.",
    "score": "사용자의 내용을 항목별로 점수화하여 평가해주세요 (10점 만점).",
    "counter": "사용자의 주장에 대해 논리적으로 반박해주세요. 다양한 관점에서 비판적으로 분석해주세요.",
}

FORMAT_INSTRUCTIONS = {
    "short": "답변은 간결하게 핵심만 3-5문장으로 작성해주세요.",
    "detailed": "답변은 상세하게 분석하여 작성해주세요. 구체적인 예시와 근거를 포함해주세요.",
}


@router.post("/message", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest):
    """
    페르소나 기반 AI 채팅 메시지 처리

    POST /api/v1/chat/message
    - 페르소나의 관점에서 사용자 메시지에 응답
    - 대화 히스토리 컨텍스트 유지
    - 파일 컨텍스트 포함 가능
    """
    try:
        llm_service = LLMService()
        persona = request.persona
        persona_name = persona.get("name", "AI 패널")

        # 시스템 프롬프트 구성
        system_parts = []

        # 페르소나 컨텍스트
        system_parts.append(f"""당신은 다음과 같은 인물입니다:
- 이름: {persona.get('name', 'AI 패널')}
- 성별: {persona.get('gender', '미지정')}
- 나이: {persona.get('age', '미지정')}
- 직업: {persona.get('occupation', '미지정')}
- 배경: {persona.get('description', '')}

이 인물의 관점과 전문성을 바탕으로 사용자에게 답변해주세요.
1인칭으로 자연스럽게 대화하되, 전문적이고 도움이 되는 답변을 제공하세요.""")

        # 목표 지시문
        if request.goal and request.goal in GOAL_INSTRUCTIONS:
            system_parts.append(GOAL_INSTRUCTIONS[request.goal])

        # 형식 지시문
        if request.format and request.format in FORMAT_INSTRUCTIONS:
            system_parts.append(FORMAT_INSTRUCTIONS[request.format])

        # 파일 컨텍스트
        if request.extracted_text:
            system_parts.append(f"""사용자가 다음 파일을 첨부했습니다 (파일명: {request.file_name or '알 수 없음'}):
---
{request.extracted_text[:5000]}
---
위 파일 내용을 참고하여 답변해주세요.""")
        elif request.file_name:
            system_parts.append(f"사용자가 '{request.file_name}' 파일을 첨부했습니다. 파일 내용을 고려하여 답변해주세요.")

        system_prompt = "\n\n".join(system_parts)

        # 대화 히스토리 구성
        messages = []
        for item in request.history[-10:]:  # 최근 10개만 포함
            messages.append({
                "role": item.role,
                "content": item.content,
            })

        # 현재 메시지 추가 (이미지 base64가 있으면 멀티모달로)
        if request.image_base64 and request.image_media_type:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": request.image_media_type, "data": request.image_base64}},
                    {"type": "text", "text": request.message},
                ],
            })
        else:
            messages.append({"role": "user", "content": request.message})

        # LLM 호출 (동기 블로킹이므로 스레드풀에서 실행)
        import asyncio
        if llm_service.client:
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, _call_chat_api, llm_service, system_prompt, messages
            )
        else:
            # Mock 모드
            response_text = _generate_mock_chat_response(
                request.message, persona_name, request.goal
            )

        return ChatMessageResponse(
            response=response_text,
            persona_name=persona_name,
        )

    except Exception as e:
        logger.error(f"Chat message failed: {e}")
        raise HTTPException(status_code=500, detail=f"채팅 응답 생성 실패: {str(e)}")


def _call_chat_api(llm_service: LLMService, system_prompt: str, messages: list[dict]) -> str:
    """Claude API를 사용하여 채팅 응답 생성"""
    import anthropic
    import time

    last_exc = None
    for attempt in range(3):
        try:
            response = llm_service.client.messages.create(
                model=llm_service.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError as e:
            last_exc = e
            if attempt < 2:
                delay = (2 ** attempt) + 0.5
                logger.warning(f"Rate limited, retrying in {delay:.1f}s")
                time.sleep(delay)
        except anthropic.APIError as e:
            last_exc = e
            if attempt < 2:
                delay = (2 ** attempt) + 0.5
                logger.warning(f"API error, retrying: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Claude API chat failed after 3 attempts: {e}")
        except Exception as e:
            logger.error(f"Unexpected chat error: {e}")
            raise

    raise last_exc


def _generate_mock_chat_response(message: str, persona_name: str, goal: str | None) -> str:
    """API 키 없을 때 Mock 응답 생성"""
    goal_responses = {
        "feedback": f"""안녕하세요, {persona_name}입니다.

말씀하신 내용을 검토해 보았습니다.

**긍정적인 부분:**
- 전체적인 구조가 잘 잡혀 있습니다
- 핵심 주장이 명확합니다

**개선이 필요한 부분:**
- 근거 자료를 보충하면 더 설득력이 있을 것 같습니다
- 결론 부분을 좀 더 강화해 보세요

더 궁금한 점이 있으시면 말씀해 주세요!""",
        "evaluate": f"""{persona_name}의 평가입니다.

**종합 평가: B+**

- 논리성: 8/10
- 창의성: 7/10
- 완성도: 7/10
- 설득력: 8/10

전반적으로 우수한 수준이며, 몇 가지 보완점을 개선하면 더 좋은 결과물이 될 것입니다.""",
        "counter": f"""{persona_name}입니다. 다른 관점에서 말씀드리겠습니다.

말씀하신 주장에는 몇 가지 재고할 부분이 있습니다:

1. 전제 조건이 충분히 검증되지 않았습니다
2. 다른 변수들을 고려하지 않은 것 같습니다
3. 반대 사례를 통해 주장을 더 탄탄하게 만들 수 있습니다

이런 관점들을 고려해보시면 어떨까요?""",
    }

    if goal and goal in goal_responses:
        return goal_responses[goal]

    return f"""안녕하세요, {persona_name}입니다!

질문해 주셔서 감사합니다.

말씀하신 "{message[:50]}..."에 대해 답변드리겠습니다.

전반적으로 좋은 접근 방식을 가지고 계시지만, 몇 가지 개선점이 있습니다.
근거 자료를 좀 더 보완하시고, 결론 부분에서 핵심 메시지를 한 번 더 강조하시면 좋겠습니다.

더 궁금한 점이 있으시면 말씀해 주세요!"""
