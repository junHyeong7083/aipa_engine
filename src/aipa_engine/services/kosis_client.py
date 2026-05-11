"""
KOSIS API 클라이언트 (통계청 공공데이터 API)

C#으로 비유하면 HttpClient로 외부 API를 호출하는 Service 클래스.
한국 통계청(KOSIS)에서 인구통계, 직업 분포 등을 가져옴.

API 문서: https://kosis.kr/openapi/devGuide/
- 통계목록 조회: statisticsList.do
- 통계자료 조회: Param/statisticsParameterData.do
"""

import json
import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

# httpx = C#의 HttpClient와 동일한 HTTP 요청 라이브러리
# (requests 보다 async 지원이 좋음)
import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class KOSISClient:
    """
    KOSIS OpenAPI 클라이언트 (C#의 public class KOSISClient : IKOSISClient)

    기능:
    1. 통계 목록 검색 (어떤 통계 테이블이 있는지)
    2. 통계 데이터 조회 (특정 테이블의 실제 숫자 데이터)
    3. 분포 데이터 조회 + 캐시 (자주 쓰는 데이터는 로컬 파일에 캐싱)
    """

    # === API 엔드포인트 URL (상수) ===
    STATISTICS_LIST_URL = "https://kosis.kr/openapi/statisticsList.do"          # 목록 조회
    STATISTICS_DATA_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"  # 데이터 조회

    # 서비스뷰 코드 (KOSIS 데이터 분류 체계)
    VIEW_CODES = {
        "subject": "MT_ZTITLE",    # 국내통계 주제별
        "org": "MT_OTITLE",        # 국내통계 기관별
        "regional": "MT_RTITLE",   # 지역통계
    }

    # === 미리 정의된 테이블 설정 (자주 쓰는 통계표의 파라미터) ===
    # KOSIS 웹사이트의 URL 생성기에서 확인한 값들
    TABLE_CONFIGS = {
        # 장래인구추계 - 성 및 연령별 추계인구 (전국)
        "population": {
            "orgId": "101",                 # 기관: 통계청
            "tblId": "DT_1BPA001",          # 테이블 ID
            "objL1": "1 ",                  # 분류1: 중위 추계
            "objL2": "0 1 2 ",              # 분류2: 0=전체, 1=남자, 2=여자
            # 분류3: 연령 코드 (5세 단위, 000=계부터 430=100세 이상까지)
            "objL3": "000 040 050 070 100 120 130 150 160 180 190 210 230 260 280 310 330 340 360 380 410 430 440 ",
            "itmId": "T10 ",                # 항목: 추계인구
            "prdSe": "Y",                   # 주기: 연간
        },
        # 경제활동인구조사 - 성/직업별 취업자
        "occupation": {
            "orgId": "101",
            "tblId": "DT_1DA7E27S_NEW",
            "objL1": "0 ",                  # 성별: 계(전체)
            "objL2": "00 10 20 30 40 50 60 70 80 90 ",  # 직업 대분류 코드
            "itmId": "T30 ",                # 항목: 취업자 수
            "prdSe": "M",                   # 주기: 월간
        },
    }

    # 직업 코드 → 직업명 매핑 (KOSIS C2 분류코드)
    OCCUPATION_CODE_MAP = {
        "10": "관리자", "20": "전문가", "30": "사무직",
        "40": "서비스직", "50": "판매직", "60": "농림어업",
        "70": "기능원", "80": "기계조작", "90": "단순노무",
    }

    # 연령 코드 → 연령 범위 매핑 (KOSIS C3 분류코드)
    AGE_CODE_MAP = {
        "040": "0-4", "050": "5-9", "070": "10-14", "100": "15-19",
        "120": "20-24", "130": "25-29", "150": "30-34", "160": "35-39",
        "180": "40-44", "190": "45-49", "210": "50-54", "230": "55-59",
        "260": "60-64", "280": "65-69", "310": "70-74", "330": "75-79",
        "340": "80-84", "360": "85-89", "380": "90-94", "410": "95-99",
        "430": "100+", "440": "age_unknown",
    }

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None, cache_days: int = 30):
        """
        생성자 (C#의 public KOSISClient(IOptions<Settings> settings))

        api_key: 직접 넣거나, 안 넣으면 .env에서 자동 로드
        cache_dir: 캐시 파일 저장 경로 (기본: data/processed/)
        cache_days: 캐시 유효 기간 (기본 30일)
        """
        if api_key:
            self.api_key = api_key
        else:
            settings = get_settings()
            self.api_key = settings.kosis_api_key

        self.cache_days = cache_days
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/processed")
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

    # ========== 통계목록 조회 API ==========

    async def list_statistics(
        self,
        view_code: str = "MT_ZTITLE",
        parent_list_id: str = "A",
    ) -> list[dict]:
        """
        통계 목록 조회 (어떤 통계표들이 있는지 검색)
        C#의 public async Task<List<StatItem>> ListStatisticsAsync(...)

        parent_list_id: A=인구, B=고용/노동, C=물가/가계 등
        """
        if not self.api_key:
            raise ValueError("KOSIS API key not configured")

        # API 요청 파라미터 조립 (C#의 new { method = "getList", ... } 같은 것)
        params = {
            "method": "getList",
            "apiKey": self.api_key,
            "vwCd": view_code,
            "parentListId": parent_list_id,
            "format": "json",
            "jsonVD": "Y",
        }

        # httpx.AsyncClient = C#의 HttpClient (비동기 HTTP 요청)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.STATISTICS_LIST_URL,
                params=params,
                timeout=30.0  # 30초 타임아웃
            )
            if response.status_code != 200:
                logger.error("KOSIS list_statistics HTTP %d: %s", response.status_code, response.text[:200])
                raise ValueError(f"KOSIS API returned HTTP {response.status_code}")
            data = response.json()       # JSON 파싱

            # KOSIS API는 에러도 200으로 오고 body에 err 키가 있음
            if isinstance(data, dict) and "err" in data:
                raise ValueError(f"KOSIS API Error: {data.get('errMsg', 'Unknown error')}")

            return data if isinstance(data, list) else []

    async def find_table(self, search_term: str) -> list[dict]:
        """
        통계표 검색 (키워드로 찾기)
        예: find_table("인구") → 인구 관련 통계표 목록 반환
        """
        results = []
        try:
            stats = await self.list_statistics(parent_list_id="A")  # 인구 카테고리 조회
            for item in stats:
                # 목록명이나 테이블명에 검색어가 포함되면 결과에 추가
                if search_term in item.get("LIST_NM", "") or search_term in item.get("TBL_NM", ""):
                    results.append(item)
        except Exception as e:
            print(f"Search failed: {e}")

        return results

    # ========== 통계자료 조회 API ==========

    async def fetch_table_data(
        self,
        org_id: str,
        tbl_id: str,
        obj_l1: str = "ALL",
        obj_l2: str = "",
        obj_l3: str = "",
        itm_id: str = "ALL",
        prd_se: str = "Y",
        start_prd_de: Optional[str] = None,
        end_prd_de: Optional[str] = None,
        new_est_prd_cnt: Optional[int] = None,
    ) -> list[dict]:
        """
        통계 데이터 조회 (실제 숫자 데이터 가져오기)
        C#의 public async Task<List<DataRow>> FetchTableDataAsync(...)

        KOSIS API의 가장 핵심 메서드.
        파라미터가 많은 이유: KOSIS 테이블마다 분류 체계가 다 다르기 때문.
        """
        if not self.api_key:
            raise ValueError("KOSIS API key not configured")

        params = {
            "method": "getList",
            "apiKey": self.api_key,
            "orgId": org_id,        # 기관 ID (101 = 통계청)
            "tblId": tbl_id,        # 통계표 ID
            "objL1": obj_l1,        # 분류1 코드
            "itmId": itm_id,        # 항목 코드
            "prdSe": prd_se,        # 주기 (Y=연간, M=월간, Q=분기)
            "format": "json",
            "jsonVD": "Y",
        }

        # 분류2, 분류3은 비어있으면 파라미터에서 제외
        if obj_l2:
            params["objL2"] = obj_l2
        if obj_l3:
            params["objL3"] = obj_l3

        # 시점 설정: newEstPrdCnt(최근 N개) 또는 start~end 범위
        if new_est_prd_cnt:
            params["newEstPrdCnt"] = str(new_est_prd_cnt)
        else:
            params["startPrdDe"] = start_prd_de or "2023"
            params["endPrdDe"] = end_prd_de or "2023"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.STATISTICS_DATA_URL,
                params=params,
                timeout=30.0
            )
            if response.status_code != 200:
                logger.error("KOSIS fetch_table_data HTTP %d: %s", response.status_code, response.text[:200])
                raise ValueError(f"KOSIS API returned HTTP {response.status_code}")
            data = response.json()

            if isinstance(data, dict) and "err" in data:
                error_msg = data.get("errMsg", "Unknown error")
                error_code = data.get("err", "")
                raise ValueError(f"KOSIS API Error [{error_code}]: {error_msg}")

            # Validate expected structure: list of dicts with DT key
            if isinstance(data, list) and data and "DT" not in data[0]:
                logger.warning("KOSIS response missing expected 'DT' key; keys present: %s", list(data[0].keys()))

            return data if isinstance(data, list) else []

    async def fetch_by_config(
        self,
        config_name: str,
        start_prd_de: Optional[str] = None,
        end_prd_de: Optional[str] = None,
        new_est_prd_cnt: Optional[int] = None,
    ) -> list[dict]:
        """
        미리 정의된 설정(TABLE_CONFIGS)으로 간편 조회
        C#의 FetchByPreset("population") 같은 편의 메서드

        예: fetch_by_config("population") → 인구 추계 데이터
            fetch_by_config("occupation") → 직업별 취업자 데이터
        """
        if config_name not in self.TABLE_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}")

        config = self.TABLE_CONFIGS[config_name]

        # start/end가 None이면 최근 1개 데이터 자동 조회
        use_new_est = new_est_prd_cnt or (start_prd_de is None and end_prd_de is None)

        return await self.fetch_table_data(
            org_id=config["orgId"],
            tbl_id=config["tblId"],
            obj_l1=config.get("objL1", "ALL"),
            obj_l2=config.get("objL2", ""),
            obj_l3=config.get("objL3", ""),
            itm_id=config.get("itmId", "ALL"),
            prd_se=config.get("prdSe", "Y"),
            start_prd_de=start_prd_de,
            end_prd_de=end_prd_de,
            new_est_prd_cnt=new_est_prd_cnt if new_est_prd_cnt else (1 if use_new_est else None),
        )

    # ========== 분포 데이터 조회 (캐시 지원) ==========
    # 자주 쓰는 분포 데이터는 로컬 JSON 파일에 캐싱해서 API 호출 최소화

    async def get_age_distribution(self, force_refresh: bool = False) -> dict[str, float]:
        """
        연령별 인구 분포 조회 (캐시 우선, 만료 시 API 호출)

        캐시 전략: 30일 이내 데이터가 있으면 캐시 사용, 없으면 API 호출
        C#의 MemoryCache/DistributedCache 패턴과 유사 (여기서는 파일 기반)
        """
        cache_file = self.cache_dir / "age_distribution.json"

        # 캐시 확인: 파일이 있고 30일 이내면 캐시 반환
        if not force_refresh and cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                cache_date = datetime.fromisoformat(cached.get("updated", "2000-01-01"))
                if (datetime.now() - cache_date).days < self.cache_days:
                    return cached["distribution"]

        # API 호출 시도
        if self.api_key:
            try:
                current_year = str(datetime.now().year)
                data = await self.fetch_by_config("population", current_year, current_year)
                distribution = self._parse_age_distribution(data)  # 원시 데이터 → 분포 변환

                if distribution:
                    # 캐시에 저장
                    cache_data = {
                        "source": "KOSIS DT_1BPA001 (장래인구추계)",
                        "updated": datetime.now().isoformat(),
                        "year": current_year,
                        "distribution": distribution,
                        "raw_record_count": len(data),
                    }
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    return distribution

            except Exception as e:
                logger.warning("Failed to fetch age distribution from KOSIS: %s", e)

        # API도 실패하면 하드코딩된 기본값 반환
        logger.info("Using hardcoded default age distribution (API unavailable or failed)")
        return self._default_age_distribution()

    async def get_gender_distribution(self, force_refresh: bool = False) -> dict[str, float]:
        """성별 인구 분포 조회 (캐시 → API → 기본값 순서)"""
        cache_file = self.cache_dir / "gender_distribution.json"

        if not force_refresh and cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                cache_date = datetime.fromisoformat(cached.get("updated", "2000-01-01"))
                if (datetime.now() - cache_date).days < self.cache_days:
                    return cached.get("distribution", self._default_gender_distribution())

        if self.api_key:
            try:
                current_year = str(datetime.now().year)
                data = await self.fetch_by_config("population", current_year, current_year)
                distribution = self._parse_gender_distribution(data)

                if distribution:
                    cache_data = {
                        "source": "KOSIS DT_1BPA001 (장래인구추계)",
                        "updated": datetime.now().isoformat(),
                        "year": current_year,
                        "distribution": distribution,
                    }
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    return distribution

            except Exception as e:
                logger.warning("Failed to fetch gender distribution from KOSIS: %s", e)

        logger.info("Using hardcoded default gender distribution (API unavailable or failed)")
        return self._default_gender_distribution()

    async def get_occupation_distribution(self, force_refresh: bool = False) -> dict[str, float]:
        """직업별 취업자 분포 조회 (캐시 → API → 기본값 순서)"""
        cache_file = self.cache_dir / "occupation_distribution.json"

        if not force_refresh and cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                cache_date = datetime.fromisoformat(cached.get("updated", "2000-01-01"))
                if (datetime.now() - cache_date).days < self.cache_days:
                    return cached.get("distribution", self._default_occupation_distribution())

        if self.api_key:
            try:
                data = await self.fetch_by_config("occupation", None, None)
                distribution = self._parse_occupation_distribution(data)

                if distribution:
                    cache_data = {
                        "source": "KOSIS DT_1DA7E27S_NEW (경제활동인구조사)",
                        "updated": datetime.now().isoformat(),
                        "distribution": distribution,
                        "raw_record_count": len(data),
                    }
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    return distribution

            except Exception as e:
                logger.warning("Failed to fetch occupation distribution from KOSIS: %s", e)

        logger.info("Using hardcoded default occupation distribution (API unavailable or failed)")
        return self._default_occupation_distribution()

    async def refresh_all(self):
        """모든 캐시 강제 새로고침 (API에서 최신 데이터 다시 받기)"""
        if not self.api_key:
            raise ValueError("KOSIS API key not configured")

        results = {
            "age": await self.get_age_distribution(force_refresh=True),
            "gender": await self.get_gender_distribution(force_refresh=True),
            "occupation": await self.get_occupation_distribution(force_refresh=True),
        }
        return results

    # ========== 데이터 파싱 (API 원시 응답 → 분포 데이터 변환) ==========

    def _parse_age_distribution(self, data: list[dict]) -> dict[str, float]:
        """
        KOSIS 장래인구추계 원시 데이터 → 연령대별 비율로 변환

        KOSIS 응답 구조:
        - C2: 성별 코드 (0=전체, 1=남자, 2=여자)
        - C3: 연령 코드 (000=계, 040=0-4세, 100=15-19세 등)
        - DT: 인구수 (실제 숫자)

        변환 로직:
        - 5세 단위 연령 코드를 10대/20대/30대 등으로 그룹핑
        - 전체(성별=0)만 사용, 계(연령=000)는 제외
        - 최종 출력: {"10대": 0.09, "20대": 0.13, ...}
        """
        if not data:
            return {}

        # 연령대별 인구수 집계용
        age_groups = {
            "10대": 0, "20대": 0, "30대": 0,
            "40대": 0, "50대": 0, "60대+": 0,
        }
        total = 0

        # KOSIS 연령 코드 → 우리 연령대 그룹 매핑
        age_code_to_group = {
            "070": "10대", "100": "10대",       # 10-14세, 15-19세
            "120": "20대", "130": "20대",       # 20-24세, 25-29세
            "150": "30대", "160": "30대",       # 30-34세, 35-39세
            "180": "40대", "190": "40대",       # 40-44세, 45-49세
            "210": "50대", "230": "50대",       # 50-54세, 55-59세
            "260": "60대+", "280": "60대+",     # 60-64세, 65-69세
            "310": "60대+", "330": "60대+",     # 70-74세, 75-79세
            "340": "60대+", "360": "60대+",     # 80-84세, 85-89세
            "380": "60대+", "410": "60대+",     # 90-94세, 95-99세
            "430": "60대+",                      # 100세 이상
        }

        # 각 데이터 행을 순회하며 집계
        for row in data:
            try:
                gender_code = row.get("C2", "")
                age_code = row.get("C3", "")
                value_str = row.get("DT", "0")
                # 쉼표 제거 후 숫자 변환 (예: "51,234" → 51234.0)
                value = float(value_str.replace(",", "")) if value_str else 0

                # 성별=전체(0)이고, 연령 코드가 매핑에 있는 경우만 집계
                if gender_code == "0" and age_code in age_code_to_group:
                    group = age_code_to_group[age_code]
                    age_groups[group] += value
                    total += value

            except (ValueError, TypeError):
                continue  # 파싱 실패한 행은 스킵

        # 비율로 변환 (인구수 → 0~1 비율)
        if total > 0:
            return {k: round(v / total, 4) for k, v in age_groups.items() if v > 0}

        return {}

    def _parse_gender_distribution(self, data: list[dict]) -> dict[str, float]:
        """
        KOSIS 장래인구추계 → 성별 비율 변환

        연령=계(000)인 행에서 남자(C2=1)/여자(C2=2)의 총인구를 추출
        """
        if not data:
            return {}

        male_total = 0
        female_total = 0

        for row in data:
            try:
                gender_code = row.get("C2", "")
                age_code = row.get("C3", "")
                value_str = row.get("DT", "0")
                value = float(value_str.replace(",", "")) if value_str else 0

                # 연령=계(000)인 경우만 성별 집계
                if age_code == "000":
                    if gender_code == "1":
                        male_total = value
                    elif gender_code == "2":
                        female_total = value

            except (ValueError, TypeError):
                continue

        total = male_total + female_total
        if total > 0:
            return {
                "male": round(male_total / total, 4),
                "female": round(female_total / total, 4),
            }

        return {}

    def _parse_occupation_distribution(self, data: list[dict]) -> dict[str, float]:
        """
        KOSIS 경제활동인구조사 → 직업별 비율 변환

        C1=성별(0=계), C2=직업코드(10=관리자, 20=전문가 등), DT=취업자수(천명)
        """
        if not data:
            return {}

        occupation_counts = {}
        total = 0

        for row in data:
            try:
                gender_code = row.get("C1", "")
                occ_code = row.get("C2", "")
                value_str = row.get("DT", "0")
                value = float(value_str.replace(",", "")) if value_str else 0

                # 성별=계(0)이고, 직업=계(00) 아닌 경우만
                if gender_code == "0" and occ_code != "00" and occ_code in self.OCCUPATION_CODE_MAP:
                    occ_name = self.OCCUPATION_CODE_MAP[occ_code]
                    occupation_counts[occ_name] = value
                    total += value

            except (ValueError, TypeError):
                continue

        if total > 0:
            return {k: round(v / total, 4) for k, v in occupation_counts.items()}

        return {}

    # ========== 기본값 (API 실패 시 Fallback) ==========

    def _default_age_distribution(self) -> dict[str, float]:
        """기본 연령 분포 (2023년 통계청 자료 기준)"""
        return {
            "10대": 0.09, "20대": 0.13, "30대": 0.14,
            "40대": 0.16, "50대": 0.17, "60대+": 0.31,
        }

    def _default_gender_distribution(self) -> dict[str, float]:
        """기본 성별 분포"""
        return {"male": 0.499, "female": 0.501}

    def _default_occupation_distribution(self) -> dict[str, float]:
        """기본 직업 분포 (2023년 경활 조사 기준)"""
        return {
            "사무직": 0.25, "서비스직": 0.18, "판매직": 0.12, "전문직": 0.15,
            "생산직": 0.10, "자영업": 0.08, "학생": 0.07, "무직/기타": 0.05,
        }
