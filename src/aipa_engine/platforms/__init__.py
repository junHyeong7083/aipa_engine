"""
SNS 플랫폼별 사용자 특성을 학습한 AI 패널 시뮬레이션 모듈

기존 AIPA 시뮬레이션과 독립적으로 동작:
- platform_data: 플랫폼별 인구/행동/특성 데이터
- platform_personas: 플랫폼 가중 페르소나 생성기
- platform_service: 플랫폼별 반응 시뮬레이션 서비스
"""

from .platform_data import PLATFORM_PROFILES, SNSPlatform
from .platform_personas import PlatformPersonaGenerator
from .platform_service import PlatformSimulationService

__all__ = [
    "PLATFORM_PROFILES",
    "SNSPlatform",
    "PlatformPersonaGenerator",
    "PlatformSimulationService",
]
