"""
네이버 데이터랩 검색어 트렌드 API 클라이언트

C#으로 비유하면 HttpClient로 네이버 OpenAPI를 호출하는 Service 클래스.
네이버 데이터랩에서 검색어 트렌드(상대 검색량)를 가져옴.

API 문서: https://developers.naver.com/docs/serviceapi/datalab/search/search.md
- 검색어 트렌드 조회: POST /v1/datalab/search
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

# httpx = C#의 HttpClient와 동일한 HTTP 요청 라이브러리
import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class NaverClient:
    """
    네이버 데이터랩 검색어 트렌드 클라이언트 (C#의 public class NaverClient : INaverClient)

    기능:
    1. 검색어 트렌드 조회 (키워드의 상대 검색량 시계열)
    2. 키워드 그룹 비교 (여러 키워드 그룹의 트렌드 비교)
    3. RAG/프롬프트용 트렌드 컨텍스트 생성
    """

    # === API 엔드포인트 URL (상수) ===
    DATALAB_SEARCH_URL = "https://openapi.naver.com/v1/datalab/search"

    # time_unit 옵션 (C#의 enum TimeUnit 같은 것)
    VALID_TIME_UNITS = {"date", "week", "month"}

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        생성자 (C#의 public NaverClient(IOptions<Settings> settings))

        client_id/client_secret: 직접 넣거나, 안 넣으면 .env에서 자동 로드
        """
        settings = get_settings()
        self.client_id = client_id or settings.naver_client_id
        self.client_secret = client_secret or settings.naver_client_secret

    # ========== 내부 헬퍼 ==========

    def _has_credentials(self) -> bool:
        """API 키가 설정되어 있는지 확인 (C#의 private bool HasCredentials())"""
        return bool(self.client_id and self.client_secret)

    def _build_headers(self) -> dict[str, str]:
        """
        HTTP 헤더 조립 (C#의 request.Headers.Add(...))

        네이버 OpenAPI는 Client-Id와 Client-Secret을 헤더로 전달
        """
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json",
        }

    # ========== 검색어 트렌드 조회 API ==========

    async def search_trend(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
        time_unit: str = "month",
    ) -> dict[str, Any]:
        """
        검색어 트렌드 조회 (키워드별 상대 검색량 시계열)
        C#의 public async Task<TrendResult> SearchTrendAsync(...)

        Args:
            keywords: 검색어 리스트 (예: ["AI", "챗봇"])
            start_date: 시작일 (yyyy-MM-dd)
            end_date: 종료일 (yyyy-MM-dd)
            time_unit: 구간 단위 ("date"=일간, "week"=주간, "month"=월간)

        Returns:
            네이버 데이터랩 응답 dict (results 안에 시계열 데이터)
        """
        if time_unit not in self.VALID_TIME_UNITS:
            raise ValueError(f"Invalid time_unit '{time_unit}'. Must be one of {self.VALID_TIME_UNITS}")

        # API 키 없으면 mock 데이터 반환 (개발/테스트 환경용)
        if not self._has_credentials():
            logger.warning("네이버 API 키 미설정 → mock 데이터 반환")
            return self._mock_search_trend(keywords, start_date, end_date, time_unit)

        # 키워드 그룹 조립: 각 키워드를 별도 그룹으로 만듦
        # 네이버 API는 keyword_groups 배열을 받음 (최대 5개 그룹)
        keyword_groups = [
            {"groupName": kw, "keywords": [kw]}
            for kw in keywords[:5]  # 최대 5개 제한
        ]

        # 요청 바디 (C#의 new { startDate = ..., endDate = ..., ... })
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": time_unit,
            "keywordGroups": keyword_groups,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.DATALAB_SEARCH_URL,
                headers=self._build_headers(),
                json=body,
                timeout=30.0,
            )
            if response.status_code != 200:
                logger.error(
                    "네이버 데이터랩 search_trend HTTP %d: %s",
                    response.status_code,
                    response.text[:300],
                )
                raise ValueError(f"Naver DataLab API returned HTTP {response.status_code}")

            data = response.json()

            # 네이버 API 에러 응답 체크
            if "errmsg" in data:
                raise ValueError(f"Naver DataLab API Error: {data['errmsg']}")

            return data

    # ========== 키워드 그룹 비교 API ==========

    async def compare_trends(
        self,
        keyword_groups: list[dict],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        time_unit: str = "month",
    ) -> dict[str, Any]:
        """
        키워드 그룹 비교 (여러 그룹의 트렌드를 한 번에 비교)
        C#의 public async Task<TrendResult> CompareTrendsAsync(...)

        Args:
            keyword_groups: 키워드 그룹 리스트
                예: [
                    {"groupName": "AI", "keywords": ["AI", "인공지능"]},
                    {"groupName": "빅데이터", "keywords": ["빅데이터", "big data"]},
                ]
            start_date: 시작일 (기본: 1년 전)
            end_date: 종료일 (기본: 오늘)
            time_unit: 구간 단위

        Returns:
            네이버 데이터랩 응답 dict
        """
        if time_unit not in self.VALID_TIME_UNITS:
            raise ValueError(f"Invalid time_unit '{time_unit}'. Must be one of {self.VALID_TIME_UNITS}")

        # 기본 날짜: 최근 1년
        now = datetime.now()
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        # API 키 없으면 mock 데이터 반환
        if not self._has_credentials():
            logger.warning("네이버 API 키 미설정 → mock 비교 데이터 반환")
            all_keywords = []
            for g in keyword_groups:
                all_keywords.append(g.get("groupName", g.get("keywords", ["unknown"])[0]))
            return self._mock_search_trend(all_keywords, start_date, end_date, time_unit)

        # 그룹 수 제한 (API 최대 5개)
        if len(keyword_groups) > 5:
            logger.warning("키워드 그룹이 5개 초과 → 처음 5개만 사용")
            keyword_groups = keyword_groups[:5]

        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": time_unit,
            "keywordGroups": keyword_groups,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.DATALAB_SEARCH_URL,
                headers=self._build_headers(),
                json=body,
                timeout=30.0,
            )
            if response.status_code != 200:
                logger.error(
                    "네이버 데이터랩 compare_trends HTTP %d: %s",
                    response.status_code,
                    response.text[:300],
                )
                raise ValueError(f"Naver DataLab API returned HTTP {response.status_code}")

            data = response.json()

            if "errmsg" in data:
                raise ValueError(f"Naver DataLab API Error: {data['errmsg']}")

            return data

    # ========== RAG/프롬프트용 컨텍스트 생성 ==========

    async def get_trending_context(self, query: str) -> str:
        """
        RAG/프롬프트 주입용 트렌드 컨텍스트 문자열 생성
        C#의 public async Task<string> GetTrendingContextAsync(string query)

        query에서 키워드를 추출하고, 최근 12개월 트렌드를 가져와서
        LLM 프롬프트에 삽입할 수 있는 형태의 한국어 요약 문자열로 반환.

        Args:
            query: 사용자 질문 또는 검색어 (예: "AI 챗봇 시장 트렌드")

        Returns:
            포맷된 트렌드 요약 문자열 (프롬프트에 바로 삽입 가능)
        """
        # 키워드 추출: 공백으로 분리, 1글자 이하 제거, 최대 3개
        raw_keywords = [w.strip() for w in query.split() if len(w.strip()) > 1]
        keywords = raw_keywords[:3] if raw_keywords else [query]

        # 최근 12개월 범위 계산
        now = datetime.now()
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        try:
            result = await self.search_trend(
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                time_unit="month",
            )
        except Exception as e:
            logger.warning("트렌드 컨텍스트 조회 실패: %s", e)
            return f"[네이버 트렌드 데이터 조회 실패: {e}]"

        # 응답 파싱 → 사람이 읽을 수 있는 요약 문자열로 변환
        return self._format_trend_context(result, keywords)

    # ========== 데이터 포맷팅 (내부 헬퍼) ==========

    def _format_trend_context(self, data: dict[str, Any], keywords: list[str]) -> str:
        """
        네이버 데이터랩 응답 → RAG 프롬프트용 한국어 요약 문자열로 변환
        C#의 private string FormatTrendContext(...)
        """
        results = data.get("results", [])
        if not results:
            return f"['{', '.join(keywords)}' 관련 네이버 검색 트렌드 데이터 없음]"

        lines = [f"## 네이버 검색 트렌드 (키워드: {', '.join(keywords)})"]

        for group in results:
            group_name = group.get("title", "unknown")
            time_data = group.get("data", [])

            if not time_data:
                lines.append(f"- {group_name}: 데이터 없음")
                continue

            # 최근 3개 시점의 수치 표시
            recent = time_data[-3:] if len(time_data) >= 3 else time_data
            trend_parts = []
            for point in recent:
                period = point.get("period", "?")
                ratio = point.get("ratio", 0)
                trend_parts.append(f"{period}: {ratio}")

            # 트렌드 방향 판단 (상승/하락/보합)
            if len(time_data) >= 2:
                first_ratio = time_data[0].get("ratio", 0)
                last_ratio = time_data[-1].get("ratio", 0)
                if last_ratio > first_ratio * 1.1:
                    direction = "상승 추세"
                elif last_ratio < first_ratio * 0.9:
                    direction = "하락 추세"
                else:
                    direction = "보합"
            else:
                direction = "데이터 부족"

            lines.append(f"- {group_name}: {direction} (최근: {', '.join(trend_parts)})")

        return "\n".join(lines)

    # ========== Mock 데이터 (API 키 없을 때 Fallback) ==========

    def _mock_search_trend(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
        time_unit: str,
    ) -> dict[str, Any]:
        """
        Mock 트렌드 데이터 생성 (API 키 미설정 시 개발/테스트용)
        C#의 FakeTrendService 같은 Fallback 구현

        실제 API 응답 형식과 동일한 구조로 가짜 데이터를 생성.
        ratio 값은 50~80 사이의 임의 패턴.
        """
        import hashlib

        logger.info("Mock 트렌드 데이터 생성 (키워드: %s)", keywords)

        results = []
        for idx, kw in enumerate(keywords):
            # 키워드별로 결정적(deterministic) 패턴 생성 (같은 키워드 → 같은 결과)
            seed = int(hashlib.md5(kw.encode()).hexdigest()[:8], 16) % 100

            # 월별 mock 데이터 생성 (최근 12개월)
            mock_data = []
            now = datetime.now()
            for m in range(12):
                date = now - timedelta(days=30 * (11 - m))
                # 결정적 변동 패턴: seed 기반으로 약간의 변동
                ratio = round(50 + (seed % 30) + (m * 1.5) + ((seed + m) % 10), 2)
                ratio = min(ratio, 100)  # 최대 100
                mock_data.append({
                    "period": date.strftime("%Y-%m-%d"),
                    "ratio": ratio,
                })

            results.append({
                "title": kw,
                "keywords": [kw],
                "data": mock_data,
            })

        return {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": time_unit,
            "results": results,
            "_mock": True,  # mock 데이터 표시 플래그
        }
