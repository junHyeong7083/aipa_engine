"""
통계 데이터 API 엔드포인트 (Statistics API Endpoints)

C#으로 비유하면 StatisticsController.cs 역할.
KOSIS(통계청) 데이터를 가져와서 프론트엔드에 제공.
캐시된 데이터가 있으면 캐시 사용, 없으면 Fallback(기본값) 반환.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# KOSIS API 클라이언트 (통계청 데이터 조회용)
from ..services.kosis_client import KOSISClient
# 네이버 데이터랩 클라이언트 (검색어 트렌드 조회용)
from ..services.naver_client import NaverClient

# C#의 ILogger<StatisticsController> 같은 것
logger = logging.getLogger(__name__)

router = APIRouter()

# === Fallback 데이터 (KOSIS 접속 실패 시 사용하는 기본값) ===
# C#의 static readonly Dictionary<string, double> 같은 상수
# 2023년 통계청 자료 기준
FALLBACK_AGE = {"10대": 0.09, "20대": 0.13, "30대": 0.14, "40대": 0.16, "50대": 0.17, "60대+": 0.31}
FALLBACK_GENDER = {"male": 0.499, "female": 0.501}
FALLBACK_OCCUPATION = {
    "사무직": 0.25, "서비스직": 0.18, "판매직": 0.12, "전문직": 0.15,
    "생산직": 0.10, "자영업": 0.08, "학생": 0.07, "무직/기타": 0.05,
}


# 응답 DTO (C#의 public class DistributionResponse { ... })
class DistributionResponse(BaseModel):
    """통계 분포 응답 (C#의 Response DTO)"""

    source: str                         # 데이터 출처 (예: "KOSIS DT_1B040M5" 또는 "fallback")
    updated_at: str                     # 마지막 업데이트 시각
    distribution: dict[str, float]      # 분포 데이터 (예: {"20대": 0.13, "30대": 0.14, ...})


# GET /api/v1/statistics/population/age
@router.get("/population/age")
async def get_age_distribution():
    """연령별 인구 분포 조회 (KOSIS 캐시 → 실패 시 Fallback)"""
    try:
        client = KOSISClient()
        dist = await client.get_age_distribution()  # KOSIS API에서 가져오기 시도
        return DistributionResponse(source="KOSIS DT_1B040M5", updated_at="cached", distribution=dist)
    except Exception:
        # KOSIS 실패 시 → 하드코딩된 기본값 사용 (서비스 중단 방지)
        logger.warning("KOSIS age data unavailable, using fallback")
        return DistributionResponse(source="fallback", updated_at="2024-01", distribution=FALLBACK_AGE)


# GET /api/v1/statistics/population/gender
@router.get("/population/gender")
async def get_gender_distribution():
    """성별 인구 분포 조회"""
    try:
        client = KOSISClient()
        dist = await client.get_gender_distribution()
        return DistributionResponse(source="KOSIS DT_1B040M1", updated_at="cached", distribution=dist)
    except Exception:
        logger.warning("KOSIS gender data unavailable, using fallback")
        return DistributionResponse(source="fallback", updated_at="2024-01", distribution=FALLBACK_GENDER)


# GET /api/v1/statistics/occupation
@router.get("/occupation")
async def get_occupation_distribution():
    """직업별 분포 조회"""
    try:
        client = KOSISClient()
        dist = await client.get_occupation_distribution()
        return DistributionResponse(source="KOSIS DT_1DA7012S", updated_at="cached", distribution=dist)
    except Exception:
        logger.warning("KOSIS occupation data unavailable, using fallback")
        return DistributionResponse(source="fallback", updated_at="2024-01", distribution=FALLBACK_OCCUPATION)


# GET /api/v1/statistics/refresh
@router.get("/refresh")
async def refresh_statistics():
    """통계 데이터 강제 새로고침 - KOSIS API에서 최신 데이터 다시 가져오기"""
    try:
        client = KOSISClient()
        await client.refresh_all()  # 모든 캐시 갱신
        return {"status": "success", "message": "Statistics refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 네이버 데이터랩 트렌드 엔드포인트 ==========


# 응답 DTO (C#의 public class TrendResponse { ... })
class TrendResponse(BaseModel):
    """검색어 트렌드 응답 (C#의 Response DTO)"""

    source: str                     # 데이터 출처 ("naver_datalab" 또는 "mock")
    keywords: list[str]             # 조회한 키워드 목록
    time_unit: str                  # 시간 단위 (date/week/month)
    results: list[dict]             # 트렌드 시계열 데이터


# GET /api/v1/statistics/trends
@router.get("/trends")
async def get_trends(
    keywords: str = Query(..., description="쉼표로 구분된 검색어 (예: AI,챗봇,빅데이터)"),
    start_date: Optional[str] = Query(None, description="시작일 (yyyy-MM-dd). 기본: 1년 전"),
    end_date: Optional[str] = Query(None, description="종료일 (yyyy-MM-dd). 기본: 오늘"),
    time_unit: str = Query("month", description="구간 단위 (date/week/month)"),
):
    """
    네이버 데이터랩 검색어 트렌드 조회
    C#의 [HttpGet("trends")] public async Task<TrendResponse> GetTrends(...)

    쉼표로 구분된 키워드의 상대 검색량 시계열 데이터를 반환.
    API 키 미설정 시 mock 데이터를 반환 (서비스 중단 방지).
    """
    keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if not keyword_list:
        raise HTTPException(status_code=400, detail="키워드를 1개 이상 입력해주세요")

    # 기본 날짜 설정
    from datetime import datetime, timedelta

    now = datetime.now()
    if not end_date:
        end_date = now.strftime("%Y-%m-%d")
    if not start_date:
        start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        client = NaverClient()
        data = await client.search_trend(
            keywords=keyword_list,
            start_date=start_date,
            end_date=end_date,
            time_unit=time_unit,
        )
        source = "mock" if data.get("_mock") else "naver_datalab"
        return TrendResponse(
            source=source,
            keywords=keyword_list,
            time_unit=time_unit,
            results=data.get("results", []),
        )
    except Exception as e:
        logger.error("트렌드 조회 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"트렌드 조회 실패: {e}")
