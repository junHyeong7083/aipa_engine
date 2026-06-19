"""
SNS 플랫폼별 시뮬레이션 API

엔드포인트:
- GET  /api/v1/platforms                  → 지원 플랫폼 목록
- GET  /api/v1/platforms/{platform_id}    → 플랫폼 상세 정보 (인구분포, 특성, 톤)
- POST /api/v1/platforms/simulate         → 플랫폼별 반응 시뮬레이션 실행
- GET  /api/v1/platforms/sessions/{sid}   → 시뮬레이션 결과 조회
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..models.survey import SurveyQuestion
from ..platforms import (
    SNSPlatform,
    PlatformSimulationService,
    PLATFORM_PROFILES,
)
from ..platforms.platform_data import list_platforms, get_profile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/platforms", tags=["platforms"])

# 싱글톤 서비스
_service = PlatformSimulationService()


# ─────────────────────────────────────
# 요청 / 응답 모델
# ─────────────────────────────────────

class PlatformSimulationRequest(BaseModel):
    platform: str = Field(..., description="유튜브/인스타그램/X/틱톡/네이버/구글/당근/디시인사이드 중 하나의 id")
    questions: list[SurveyQuestion] = Field(..., min_length=1)
    panel_size: int = Field(30, ge=5, le=200, description="가상 패널 크기")


class PlatformInfoResponse(BaseModel):
    id: str
    name: str
    description: str
    age_distribution: dict[str, float]
    gender_ratio: dict[str, float]
    dominant_occupations: list[str]
    platform_traits: list[str]
    tone_guide: str
    common_topics: list[str]
    sources: list[str]


# ─────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────

@router.get("/")
async def get_platforms():
    """지원하는 SNS 플랫폼 목록 반환 (앱 선택 화면용)"""
    return {"platforms": list_platforms()}


@router.get("/{platform_id}")
async def get_platform_info(platform_id: str) -> PlatformInfoResponse:
    """특정 플랫폼의 사용자 특성 데이터 상세 조회"""
    try:
        platform = SNSPlatform(platform_id.lower())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 플랫폼: {platform_id}")

    profile = get_profile(platform)
    return PlatformInfoResponse(
        id=platform.value,
        name=profile.name_kr,
        description=profile.description,
        age_distribution=profile.age_distribution,
        gender_ratio=profile.gender_ratio,
        dominant_occupations=profile.dominant_occupations,
        platform_traits=profile.platform_traits,
        tone_guide=profile.tone_guide,
        common_topics=profile.common_topics,
        sources=profile.sources,
    )


@router.post("/simulate")
async def simulate(request: PlatformSimulationRequest):
    """플랫폼별 사용자 반응 시뮬레이션 실행 (동기 - 30~60초 이내)"""
    try:
        platform = SNSPlatform(request.platform.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 플랫폼: {request.platform}")

    try:
        result = await _service.run(
            platform=platform,
            questions=request.questions,
            panel_size=request.panel_size,
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Platform simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """저장된 시뮬레이션 결과 조회"""
    result = _service.get(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    return result.to_dict()
