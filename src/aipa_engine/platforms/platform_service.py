"""
SNS 플랫폼별 시뮬레이션 서비스

기존 simulation_service와 별개로 동작:
1. 플랫폼 선택 → 플랫폼 가중 페르소나 생성
2. 기존 ResponseGenerator로 응답 생성하되, 플랫폼 traits가 자동 적용됨
3. 플랫폼 톤 가이드를 시뮬레이션 메타데이터에 첨부
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from ..models.persona import Persona
from ..models.survey import SurveyQuestion, SurveyResponse
from ..services.response_generator import ResponseGenerator
from .platform_data import SNSPlatform, get_profile, PlatformProfile
from .platform_personas import PlatformPersonaGenerator

logger = logging.getLogger(__name__)


class PlatformSimulationResult:
    def __init__(self, session_id: str, platform: SNSPlatform):
        self.session_id = session_id
        self.platform = platform
        self.profile: PlatformProfile = get_profile(platform)
        self.personas: list[Persona] = []
        self.responses: list[SurveyResponse] = []
        self.response_distribution: dict[str, dict[str, float]] = {}
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "platform": self.platform.value,
            "platform_name": self.profile.name_kr,
            "platform_traits": self.profile.platform_traits,
            "common_topics": self.profile.common_topics,
            "tone_guide": self.profile.tone_guide,
            "personas": [p.model_dump() for p in self.personas],
            "responses": [r.model_dump() for r in self.responses],
            "response_distribution": self.response_distribution,
            "panel_size": len(self.personas),
            "created_at": self.created_at,
            "sources": self.profile.sources,
        }


class PlatformSimulationService:
    """SNS 플랫폼별 반응 시뮬레이션"""

    def __init__(self):
        self._sessions: dict[str, PlatformSimulationResult] = {}
        self._response_generator: Optional[ResponseGenerator] = None

    @property
    def response_generator(self) -> ResponseGenerator:
        if self._response_generator is None:
            self._response_generator = ResponseGenerator()
        return self._response_generator

    async def run(
        self,
        platform: SNSPlatform | str,
        questions: list[SurveyQuestion],
        panel_size: int = 30,
    ) -> PlatformSimulationResult:
        """
        플랫폼 시뮬레이션 실행

        Args:
            platform: 대상 SNS 플랫폼
            questions: 평가할 질문 리스트
            panel_size: 가상 패널 크기

        Returns:
            플랫폼별 응답 분포 + 개별 응답
        """
        if isinstance(platform, str):
            platform = SNSPlatform(platform)

        session_id = str(uuid.uuid4())
        result = PlatformSimulationResult(session_id, platform)

        # 1. 플랫폼 가중 페르소나 생성
        generator = PlatformPersonaGenerator(platform)
        result.personas = generator.generate_panel(panel_size)
        logger.info(
            f"[Platform {platform.value}] {panel_size}명 페르소나 생성 완료"
        )

        # 2. 각 질문에 대해 모든 페르소나가 응답
        for question in questions:
            choice_counts: dict[str, int] = {c: 0 for c in question.choices}

            for persona in result.personas:
                try:
                    response = await self.response_generator.generate_response(
                        persona=persona,
                        question=question,
                        generate_explanation=False,
                    )
                    result.responses.append(response)

                    selected = response.selected_choice
                    if selected and selected in choice_counts:
                        choice_counts[selected] += 1

                except Exception as e:
                    logger.warning(f"응답 생성 실패: {e}")
                    continue

            # 분포 계산
            total = sum(choice_counts.values())
            if total > 0:
                result.response_distribution[question.id] = {
                    c: round(n / total, 4) for c, n in choice_counts.items()
                }
            else:
                result.response_distribution[question.id] = {c: 0.0 for c in question.choices}

        self._sessions[session_id] = result
        logger.info(f"[Platform {platform.value}] 시뮬레이션 완료: {session_id}")
        return result

    def get(self, session_id: str) -> Optional[PlatformSimulationResult]:
        return self._sessions.get(session_id)
