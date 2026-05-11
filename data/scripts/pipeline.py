"""
AIPA 데이터 자동 수집 파이프라인

수집 소스:
    1. KOSIS (통계청) - 인구, 성별, 직업 분포
    2. 공공데이터포털 (data.go.kr) - 인구이동, 혼인/이혼
    3. 네이버 데이터랩 - 검색 트렌드, 쇼핑 인사이트

사용법:
    python data/scripts/pipeline.py                    # 전체 수집
    python data/scripts/pipeline.py --source kosis     # KOSIS만
    python data/scripts/pipeline.py --source naver     # 네이버만
    python data/scripts/pipeline.py --source data_kr   # 공공데이터만

스케줄 등록 (Windows Task Scheduler):
    schtasks /create /tn "AIPA-Pipeline" /tr "python C:\\...\\pipeline.py" /sc daily /st 03:00

스케줄 등록 (Linux cron):
    0 3 * * * cd /path/to/AIPA_Engine && python data/scripts/pipeline.py >> data/pipeline/pipeline.log 2>&1
"""

import asyncio
import json
import logging
import argparse
import os
import sys
# ABC, abstractmethod = C#의 abstract class, abstract method 와 동일
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import random

# .env 파일에서 환경변수 로드 (C#의 ConfigurationBuilder().AddEnvironmentVariables())
from dotenv import load_dotenv
load_dotenv()

# httpx = C#의 HttpClient (비동기 HTTP 요청 라이브러리)
import httpx

# Firestore 연동 (선택적 - 초기화 실패해도 로컬 파일 저장은 계속 동작)
_firestore = None
try:
    # AIPA Engine 패키지가 설치되어 있으면 Firestore 서비스 사용
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from aipa_engine.services.firestore_service import FirestoreService
    _firestore = FirestoreService()
    if _firestore.available:
        logging.getLogger("aipa.pipeline").info("Firestore connected")
    else:
        _firestore = None
except Exception:
    pass  # Firestore 없어도 로컬 저장으로 정상 동작

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

# __file__ = 현재 파일 경로. resolve() = 절대경로로 변환
# parent.parent = data/scripts/ → data/ (2단계 올라감)
BASE_DIR = Path(__file__).resolve().parent.parent  # data/ 폴더
PIPELINE_DIR = BASE_DIR / "pipeline" / "daily"     # 수집 데이터 저장 경로
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)    # 폴더 없으면 자동 생성

LOG_FILE = BASE_DIR / "pipeline" / "pipeline.log"  # 로그 파일 경로

# 로깅 설정 (콘솔 + 파일 동시 출력)
# C#의 ILogger + Serilog 같은 역할
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),                                    # 콘솔 출력
        logging.FileHandler(LOG_FILE, encoding="utf-8"),           # 파일 출력
    ],
)
logger = logging.getLogger("aipa.pipeline")


class APILimitExhausted(Exception):
    """API 일일 호출 한도 초과 예외"""
    pass


async def retry_async(coro_func, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
    """
    비동기 함수를 지수 백오프로 재시도.
    coro_func: 호출할 async 함수 (코루틴 팩토리)
    max_retries: 최대 재시도 횟수
    base_delay: 첫 번째 대기 시간 (초)
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except APILimitExhausted:
            raise  # API 한도 초과는 재시도하지 않음
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
    raise last_exc


def today_str() -> str:
    """오늘 날짜 문자열 반환 (예: "2026-03-15")"""
    return datetime.now().strftime("%Y-%m-%d")


def save_dataset(source: str, name: str, data: dict):
    """
    수집된 데이터를 파일로 저장하는 유틸 함수

    2가지 형태로 저장:
    1. 날짜별 JSON 파일: search_trend_2026-03-15.json (매일 덮어씀)
    2. 누적 JSONL 파일: search_trend_history.jsonl (매일 한 줄씩 추가 = 시계열 데이터)

    C#의 File.WriteAllText() + File.AppendAllText() 패턴
    """
    date = today_str()
    dir_path = PIPELINE_DIR / source                     # 예: data/pipeline/daily/naver/
    dir_path.mkdir(parents=True, exist_ok=True)

    # 1. 날짜별 JSON 파일 (오늘 데이터 = 스냅샷)
    file_path = dir_path / f"{name}_{date}.json"
    record = {
        "collected_at": datetime.now().isoformat(),      # 수집 시각
        "source": source,                                 # 출처 (kosis, naver 등)
        "name": name,                                     # 데이터 이름
        **data,                                           # 실제 데이터 (spread operator)
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # 2. 누적 JSONL 파일 (히스토리 = 시계열)
    # JSONL = 한 줄에 JSON 하나씩. 매일 추가되므로 시간이 지날수록 데이터가 쌓임
    # 중복 방지: 같은 날짜의 엔트리가 이미 있으면 추가하지 않음
    history_path = dir_path / f"{name}_history.jsonl"
    already_exists = False
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing = json.loads(line)
                    if existing.get("collected_at", "")[:10] == date:
                        already_exists = True
                        break
        except (json.JSONDecodeError, OSError):
            pass  # 파일 손상 시 무시하고 추가 진행

    if not already_exists:
        with open(history_path, "a", encoding="utf-8") as f:      # "a" = append 모드
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        logger.debug(f"  Skipped duplicate history entry for {name} on {date}")

    logger.info(f"  Saved: {file_path.name} ({len(json.dumps(data))} bytes)")

    # 3. Firestore에도 저장 (연결되어 있으면)
    if _firestore:
        _firestore.save_pipeline_data(source, name, data)


# ─────────────────────────────────────────────
# Base Collector (추상 클래스)
# ─────────────────────────────────────────────

class BaseCollector(ABC):
    """
    데이터 수집기 베이스 클래스 (C#의 abstract class BaseCollector)

    모든 수집기(KOSIS, 네이버 등)는 이 클래스를 상속받아서
    collect() 메서드를 구현해야 함.
    C#의 Template Method 패턴.
    """

    name: str = "base"  # 수집기 이름 (하위 클래스에서 오버라이드)

    @abstractmethod
    async def collect(self) -> list[dict]:
        """
        데이터 수집 실행 (하위 클래스에서 구현 필수)
        C#의 public abstract Task<List<DataItem>> CollectAsync();

        반환 형식: [{"name": "dataset_name", "data": {...}}, ...]
        """
        pass

    async def run(self):
        """
        수집 + 저장 실행 (Template Method)
        C#의 public async Task RunAsync() - 상속받은 클래스에서 override 불필요

        collect()로 데이터 수집 → save_dataset()으로 저장
        """
        logger.info(f"[{self.name}] 수집 시작...")
        try:
            results = await self.collect()                     # 하위 클래스의 collect() 호출
            for result in results:
                save_dataset(self.name, result["name"], result["data"])  # 각 데이터셋 저장
            logger.info(f"[{self.name}] 완료: {len(results)}개 데이터셋 저장")
            return results
        except Exception as e:
            logger.error(f"[{self.name}] 실패: {e}")
            return []


# ─────────────────────────────────────────────
# 1. KOSIS 수집기 (통계청)
# ─────────────────────────────────────────────

class KOSISCollector(BaseCollector):
    """
    통계청 KOSIS OpenAPI 수집기 (C#의 public class KOSISCollector : BaseCollector)

    5개 통계표를 수집:
    - population_age: 연령별 인구 (장래인구추계)
    - population_gender: 성별 인구
    - occupation: 직업별 취업자 (경제활동인구조사)
    - household_income: 가구소득 분위 (가계동향조사)
    - consumer_price: 소비자물가지수
    """

    name = "kosis"

    STATISTICS_DATA_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

    # 수집할 테이블 목록 (각 테이블의 KOSIS API 파라미터)
    # C#의 Dictionary<string, TableConfig> 같은 것
    TABLES = {
        "population_age": {
            "description": "연령별 인구 (주민등록 인구통계)",
            "orgId": "110",                 # 기관: 행정안전부
            "tblId": "DT_1B040A3",          # 주민등록 연령별 인구
            "objL1": "ALL",                 # 전체 지역
            "itmId": "ALL",                 # 전체 항목
            "prdSe": "M",                   # 월간
        },
        "population_gender": {
            "description": "성별 인구 (장래인구추계)",
            "orgId": "101",
            "tblId": "DT_1BPA001",
            "objL1": "1+",
            "objL2": "0+1+2+",              # 전체, 남자, 여자
            "objL3": "000+",                 # 계(합계)만
            "itmId": "T10+",
            "prdSe": "Y",
        },
        "occupation": {
            "description": "직업별 취업자 (경제활동인구조사)",
            "orgId": "101",
            "tblId": "DT_1DA7002S",         # 직업별 취업자 (간이분류)
            "objL1": "ALL",                 # 전체
            "itmId": "ALL",                 # 전체 항목
            "prdSe": "M",                   # 월간
        },
        "household_income": {
            "description": "가구소득 분위별 소득 (가계동향조사)",
            "orgId": "101",
            "tblId": "DT_1L9H002",
            "objL1": "ALL",                 # 전체
            "itmId": "ALL",                 # 전체 항목
            "prdSe": "Q",                   # 분기
        },
        "consumer_price": {
            "description": "소비자물가지수 (총지수)",
            "orgId": "101",
            "tblId": "DT_1J20001",          # 소비자물가지수 총지수
            "objL1": "ALL",                 # 전체
            "itmId": "ALL",                 # 전체 항목
            "prdSe": "M",                   # 월간
        },
    }

    def __init__(self):
        # .env에서 KOSIS API 키 로드
        self.api_key = os.environ.get("KOSIS_API_KEY", "")

    async def _fetch_table(self, table_key: str, config: dict) -> dict:
        """
        KOSIS API에서 단일 테이블 데이터 조회
        C#의 private async Task<List<DataRow>> FetchTableAsync(...)
        """
        # API 파라미터 조립
        params = {
            "method": "getList",
            "apiKey": self.api_key,
            "orgId": config["orgId"],
            "tblId": config["tblId"],
            "objL1": config.get("objL1", "ALL"),
            "itmId": config.get("itmId", "ALL"),
            "prdSe": config.get("prdSe", "Y"),
            "newEstPrdCnt": "1",            # 최근 1개 시점만 조회
            "format": "json",
            "jsonVD": "Y",
        }
        # 분류2, 분류3이 있으면 추가
        if config.get("objL2"):
            params["objL2"] = config["objL2"]
        if config.get("objL3"):
            params["objL3"] = config["objL3"]

        # HTTP GET 요청 (C#의 HttpClient.GetAsync())
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(self.STATISTICS_DATA_URL, params=params)
            resp.raise_for_status()         # HTTP 에러면 예외
            data = resp.json()

        # KOSIS는 에러도 200으로 반환하고 body에 err 키가 있음
        if isinstance(data, dict) and "err" in data:
            raise ValueError(f"KOSIS Error [{data.get('err')}]: {data.get('errMsg')}")

        return data if isinstance(data, list) else []

    async def collect(self) -> list[dict]:
        """
        모든 테이블 순회하며 데이터 수집 (BaseCollector.collect 구현)
        실패한 테이블은 스킵하고 성공한 것만 반환
        """
        if not self.api_key:
            logger.warning("[kosis] KOSIS_API_KEY not set, skipping")
            return []

        results = []
        for table_key, config in self.TABLES.items():
            try:
                raw_data = await retry_async(self._fetch_table, table_key, config)
                # 기본 응답 검증: 데이터가 리스트인지 확인
                if not isinstance(raw_data, list):
                    logger.error(f"  [kosis] {table_key}: unexpected response type {type(raw_data)}")
                    continue
                results.append({
                    "name": table_key,
                    "data": {
                        "description": config["description"],
                        "table_id": config["tblId"],
                        "record_count": len(raw_data),
                        "records": raw_data,            # 원시 데이터 그대로 저장
                    },
                })
                logger.info(f"  [kosis] {table_key}: {len(raw_data)} records")
            except Exception as e:
                logger.error(f"  [kosis] {table_key} failed: {e}")

        return results


# ─────────────────────────────────────────────
# 2. 공공데이터포털 수집기
# ─────────────────────────────────────────────

class DataKRCollector(BaseCollector):
    """
    공공데이터포털 수집기 (C#의 public class DataKRCollector : BaseCollector)

    현재는 KOSIS API를 통해 추가 통계 수집:
    - population_move: 국내인구이동 (시도별 전입/전출)
    - marriage_divorce: 혼인/이혼 건수 (월별)
    """

    name = "data_kr"

    # 수집할 API 목록
    APIS = {
        "population_move": {
            "description": "국내인구이동통계 (시도별 전입/전출)",
            "url": "https://kosis.kr/openapi/Param/statisticsParameterData.do",
            "params": {
                "orgId": "101",
                "tblId": "DT_1B26001",
                "objL1": "ALL",
                "itmId": "T10",
                "prdSe": "M",               # 월간
                "newEstPrdCnt": "1",
            },
        },
        "marriage_divorce": {
            "description": "혼인/이혼 건수 (월별)",
            "url": "https://kosis.kr/openapi/Param/statisticsParameterData.do",
            "params": {
                "orgId": "101",
                "tblId": "DT_1B83A05",
                "objL1": "ALL",
                "itmId": "ALL",
                "prdSe": "M",
                "newEstPrdCnt": "1",
            },
        },
    }

    def __init__(self):
        self.kosis_key = os.environ.get("KOSIS_API_KEY", "")
        self.data_kr_key = os.environ.get("DATA_KR_API_KEY", "")  # 추후 공공데이터포털 키

    async def collect(self) -> list[dict]:
        """각 API를 순회하며 데이터 수집"""
        results = []
        for api_key, config in self.APIS.items():
            try:
                async def _fetch_data_kr(url, params):
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.get(url, params=params)
                        resp.raise_for_status()
                        return resp.json()

                # 파라미터에 API 키 추가
                params = {**config["params"], "apiKey": self.kosis_key, "method": "getList", "format": "json", "jsonVD": "Y"}
                data = await retry_async(_fetch_data_kr, config["url"], params)

                if isinstance(data, dict) and "err" in data:
                    logger.error(f"  [data_kr] {api_key}: API error {data.get('errMsg')}")
                    continue

                records = data if isinstance(data, list) else []
                results.append({
                    "name": api_key,
                    "data": {
                        "description": config["description"],
                        "record_count": len(records),
                        "records": records,
                    },
                })
                logger.info(f"  [data_kr] {api_key}: {len(records)} records")
            except Exception as e:
                logger.error(f"  [data_kr] {api_key} failed: {e}")

        return results


# ─────────────────────────────────────────────
# 3. 네이버 데이터랩 수집기
# ─────────────────────────────────────────────

class NaverDataLabCollector(BaseCollector):
    """
    네이버 데이터랩 API 수집기 (C#의 public class NaverDataLabCollector : BaseCollector)

    2가지 데이터 수집:
    1. 검색어 트렌드: 소비 관련 키워드 5그룹의 일별 검색량 비율
    2. 쇼핑 인사이트: 6개 쇼핑 카테고리의 일별 트렌드

    필요 환경변수:
        NAVER_CLIENT_ID      (네이버 개발자센터에서 발급)
        NAVER_CLIENT_SECRET  (같이 발급됨)
    """

    name = "naver"

    # API 엔드포인트
    SEARCH_TREND_URL = "https://openapi.naver.com/v1/datalab/search"
    SHOPPING_TREND_URL = "https://openapi.naver.com/v1/datalab/shopping/categories"

    # 소비 트렌드 추적용 검색 키워드 그룹 (API 1회당 최대 5그룹)
    # 여러 세트로 나눠서 호출 → 일일 한도(1000건) 최대 활용
    SEARCH_GROUP_SETS = [
        # --- 세트 1: 소비 대분류 ---
        [
            {"groupName": "건강식품", "keywords": ["건강식품", "영양제", "프로바이오틱스", "비타민"]},
            {"groupName": "패션", "keywords": ["패션", "옷", "신발", "가방"]},
            {"groupName": "가전제품", "keywords": ["가전", "냉장고", "에어컨", "TV"]},
            {"groupName": "뷰티", "keywords": ["화장품", "스킨케어", "립틴트", "선크림"]},
            {"groupName": "식품배달", "keywords": ["배달", "배달음식", "배달앱", "쿠팡이츠"]},
        ],
        # --- 세트 2: MZ세대 소비 ---
        [
            {"groupName": "명품", "keywords": ["명품", "구찌", "샤넬", "루이비통"]},
            {"groupName": "카페", "keywords": ["카페", "스타벅스", "커피", "디저트"]},
            {"groupName": "여행", "keywords": ["여행", "항공권", "호텔", "해외여행"]},
            {"groupName": "OTT", "keywords": ["넷플릭스", "유튜브프리미엄", "디즈니플러스", "웨이브"]},
            {"groupName": "자기계발", "keywords": ["자기계발", "온라인강의", "독서", "영어공부"]},
        ],
        # --- 세트 3: 생활 필수 ---
        [
            {"groupName": "육아", "keywords": ["육아", "기저귀", "분유", "아기옷"]},
            {"groupName": "반려동물", "keywords": ["반려동물", "강아지", "고양이", "펫푸드"]},
            {"groupName": "인테리어", "keywords": ["인테리어", "가구", "이케아", "조명"]},
            {"groupName": "자동차", "keywords": ["자동차", "전기차", "중고차", "테슬라"]},
            {"groupName": "보험재테크", "keywords": ["보험", "적금", "주식", "부동산"]},
        ],
        # --- 세트 4: 식품 세분화 ---
        [
            {"groupName": "건강음료", "keywords": ["프로틴", "제로음료", "콤부차", "녹즙"]},
            {"groupName": "밀키트", "keywords": ["밀키트", "간편식", "냉동식품", "새벽배송"]},
            {"groupName": "외식", "keywords": ["맛집", "레스토랑", "회식", "브런치"]},
            {"groupName": "다이어트", "keywords": ["다이어트", "단백질", "샐러드", "저탄수화물"]},
            {"groupName": "간식", "keywords": ["과자", "아이스크림", "초콜릿", "편의점간식"]},
        ],
        # --- 세트 5: 테크/디지털 ---
        [
            {"groupName": "스마트폰", "keywords": ["아이폰", "갤럭시", "스마트폰", "폴드"]},
            {"groupName": "노트북", "keywords": ["노트북", "맥북", "아이패드", "태블릿"]},
            {"groupName": "게임", "keywords": ["게임", "닌텐도", "PS5", "스팀"]},
            {"groupName": "AI서비스", "keywords": ["ChatGPT", "AI", "인공지능", "클로바"]},
            {"groupName": "이어폰", "keywords": ["에어팟", "이어폰", "헤드폰", "버즈"]},
        ],
        # --- 세트 6: 뷰티 세분화 ---
        [
            {"groupName": "기초화장품", "keywords": ["토너", "세럼", "크림", "앰플"]},
            {"groupName": "색조화장품", "keywords": ["립스틱", "파운데이션", "아이섀도", "마스카라"]},
            {"groupName": "헤어케어", "keywords": ["샴푸", "탈모", "염색", "헤어트리트먼트"]},
            {"groupName": "남성화장품", "keywords": ["남성화장품", "면도기", "남성스킨케어", "향수"]},
            {"groupName": "네일아트", "keywords": ["네일", "젤네일", "네일아트", "매니큐어"]},
        ],
        # --- 세트 7: 패션 세분화 ---
        [
            {"groupName": "스포츠웨어", "keywords": ["나이키", "아디다스", "뉴발란스", "운동화"]},
            {"groupName": "아우터", "keywords": ["패딩", "코트", "자켓", "바람막이"]},
            {"groupName": "럭셔리", "keywords": ["에르메스", "프라다", "디올", "발렌시아가"]},
            {"groupName": "캐주얼", "keywords": ["무신사", "유니클로", "자라", "에이블리"]},
            {"groupName": "주얼리", "keywords": ["반지", "목걸이", "귀걸이", "시계"]},
        ],
        # --- 세트 8: 라이프스타일 ---
        [
            {"groupName": "운동", "keywords": ["헬스장", "필라테스", "요가", "크로스핏"]},
            {"groupName": "캠핑", "keywords": ["캠핑", "텐트", "캠핑용품", "글램핑"]},
            {"groupName": "골프", "keywords": ["골프", "골프웨어", "골프장", "스크린골프"]},
            {"groupName": "결혼", "keywords": ["웨딩", "결혼준비", "스드메", "신혼여행"]},
            {"groupName": "이사", "keywords": ["이사", "원룸", "전세", "월세"]},
        ],
        # --- 세트 9: 교육 ---
        [
            {"groupName": "학원", "keywords": ["학원", "과외", "입시", "수능"]},
            {"groupName": "어학", "keywords": ["토익", "토플", "영어회화", "일본어"]},
            {"groupName": "자격증", "keywords": ["자격증", "공무원", "취업", "코딩"]},
            {"groupName": "유아교육", "keywords": ["유치원", "어린이집", "영어유치원", "학습지"]},
            {"groupName": "대학교", "keywords": ["대학교", "편입", "대학원", "유학"]},
        ],
        # --- 세트 10: 계절/이벤트 ---
        [
            {"groupName": "여름", "keywords": ["수영복", "선크림", "에어컨", "빙수"]},
            {"groupName": "겨울", "keywords": ["패딩", "히터", "스키", "핫초코"]},
            {"groupName": "명절", "keywords": ["설날", "추석", "선물세트", "한복"]},
            {"groupName": "블프", "keywords": ["블랙프라이데이", "세일", "할인", "쿠폰"]},
            {"groupName": "밸런타인", "keywords": ["밸런타인", "화이트데이", "기념일", "선물"]},
        ],
    ]

    # 하위 호환 (기존 코드용)
    SEARCH_GROUPS = SEARCH_GROUP_SETS[0]

    # 쇼핑 카테고리 코드 (네이버 쇼핑 인사이트 API용) — 대폭 확장
    SHOPPING_CATEGORIES = {
        # 대분류
        "fashion": "50000000",          # 패션의류
        "beauty": "50000002",           # 화장품/미용
        "food": "50000006",             # 식품
        "digital": "50000001",          # 디지털/가전
        "living": "50000004",           # 생활/건강
        "sports": "50000007",           # 스포츠/레저
        # 세분류 추가
        "furniture": "50000003",        # 가구/인테리어
        "baby": "50000005",             # 출산/육아
        "book": "50000008",             # 도서
        "pet": "50000010",              # 반려동물용품
        "travel": "50000011",           # 여행/문화
        "car": "50000009",              # 자동차용품
        "fashion_acc": "50000013",      # 패션잡화
        "kids_fashion": "50000012",     # 아동/유아패션
    }

    def __init__(self):
        # 네이버 개발자센터에서 발급받은 인증 정보
        self.client_id = os.environ.get("NAVER_CLIENT_ID", "")
        self.client_secret = os.environ.get("NAVER_CLIENT_SECRET", "")

    def _headers(self) -> dict:
        """네이버 API 인증 헤더 (C#의 HttpRequestHeaders에 추가하는 것과 동일)"""
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json",
        }

    async def _fetch_search_trend(self) -> dict:
        """기본 세트(세트1)로 검색어 트렌드 조회"""
        return await self._fetch_search_trend_set(self.SEARCH_GROUPS)

    async def _fetch_search_trend_set(self, keyword_groups: list) -> dict:
        """
        검색어 트렌드 조회 (키워드 그룹 세트 지정)
        반환: 각 키워드 그룹별 일별 검색량 비율 (0~100)
        """
        today = datetime.now()
        start = today.replace(day=1).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")

        body = {
            "startDate": start,
            "endDate": end,
            "timeUnit": "date",
            "keywordGroups": keyword_groups,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.SEARCH_TREND_URL,
                headers=self._headers(),
                json=body,
            )
            if resp.status_code == 429:
                raise APILimitExhausted("검색어트렌드 API 일일 한도 소진")
            resp.raise_for_status()
            return resp.json()

    async def _fetch_shopping_trend(self, cat_name: str, cat_code: str) -> dict:
        """
        쇼핑 카테고리 트렌드 조회
        반환: 해당 카테고리의 일별 쇼핑 검색량 비율 (0~100)
        """
        today = datetime.now()
        start = today.replace(day=1).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")

        body = {
            "startDate": start,
            "endDate": end,
            "timeUnit": "date",
            "category": [{"name": cat_name, "param": [cat_code]}],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.SHOPPING_TREND_URL,
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            return resp.json()

    async def collect(self) -> list[dict]:
        """
        검색 트렌드 + 쇼핑 트렌드 전부 수집

        확장된 수집:
        - 검색 트렌드: 10세트 × 5그룹 = 50개 키워드 그룹 (API 10회)
        - 쇼핑 인사이트: 14개 카테고리 (API 14회)
        - 총 API 호출: 약 24회 / 일일 한도 2,000회
        """
        if not self.client_id or not self.client_secret:
            logger.warning("[naver] NAVER_CLIENT_ID/SECRET not set, skipping")
            return []

        results = []

        # 1. 검색 트렌드 (세트별로 순회 → 각 세트가 1 API 호출)
        for i, group_set in enumerate(self.SEARCH_GROUP_SETS):
            set_name = f"search_trend_{i+1:02d}"
            try:
                trend_data = await retry_async(self._fetch_search_trend_set, group_set)
                # 기본 응답 검증
                if not isinstance(trend_data, dict) or "results" not in trend_data:
                    logger.error(f"  [naver] {set_name}: unexpected response format")
                    continue
                results.append({
                    "name": set_name,
                    "data": {
                        "description": f"네이버 검색어 트렌드 세트{i+1}",
                        "groups": [g["groupName"] for g in group_set],
                        "results": trend_data.get("results", []),
                    },
                })
                logger.info(f"  [naver] {set_name}: {len(trend_data.get('results', []))} groups")
            except APILimitExhausted as e:
                logger.warning(f"  [naver] {set_name}: {e} - stopping search trend collection")
                break
            except Exception as e:
                logger.error(f"  [naver] {set_name} failed: {e}")

        # 2. 쇼핑 카테고리 트렌드 (14개 카테고리 순회)
        for cat_name, cat_code in self.SHOPPING_CATEGORIES.items():
            try:
                shop_data = await retry_async(self._fetch_shopping_trend, cat_name, cat_code)
                # 기본 응답 검증
                if not isinstance(shop_data, dict) or "results" not in shop_data:
                    logger.error(f"  [naver] shopping_{cat_name}: unexpected response format")
                    continue
                results.append({
                    "name": f"shopping_{cat_name}",
                    "data": {
                        "description": f"네이버 쇼핑 트렌드: {cat_name}",
                        "category_code": cat_code,
                        "results": shop_data.get("results", []),
                    },
                })
                logger.info(f"  [naver] shopping_{cat_name}: OK")
            except Exception as e:
                logger.error(f"  [naver] shopping_{cat_name} failed: {e}")

        return results


# ─────────────────────────────────────────────
# 파이프라인 실행 (메인 로직)
# ─────────────────────────────────────────────

# 사용 가능한 수집기 목록 (C#의 Dictionary<string, Type> 같은 것)
COLLECTORS = {
    "kosis": KOSISCollector,
    "data_kr": DataKRCollector,
    "naver": NaverDataLabCollector,
}


async def run_pipeline(sources: list[str] | None = None):
    """
    전체 파이프라인 실행 (C#의 public async Task RunPipelineAsync())

    sources가 None이면 전체 수집기 실행, 지정하면 해당 수집기만 실행
    """
    logger.info(f"{'='*50}")
    logger.info(f"AIPA Data Pipeline - {today_str()}")
    logger.info(f"{'='*50}")

    targets = sources or list(COLLECTORS.keys())  # 소스 미지정 시 전체
    total_datasets = 0

    for source_name in targets:
        if source_name not in COLLECTORS:
            logger.warning(f"Unknown source: {source_name}")
            continue

        collector = COLLECTORS[source_name]()   # 수집기 인스턴스 생성
        results = await collector.run()          # 수집 + 저장 실행
        total_datasets += len(results)

    logger.info(f"{'='*50}")
    logger.info(f"Pipeline complete: {total_datasets} datasets saved")
    logger.info(f"Output: {PIPELINE_DIR}")
    logger.info(f"{'='*50}")

    return total_datasets


def main():
    """
    CLI 진입점 (C#의 static void Main(string[] args))

    커맨드라인 인자 파싱:
    --source kosis  : KOSIS만 수집
    --list          : 수집 가능한 데이터 목록 출력
    """
    parser = argparse.ArgumentParser(description="AIPA 데이터 수집 파이프라인")
    parser.add_argument(
        "--source", type=str, choices=list(COLLECTORS.keys()),
        help="특정 소스만 수집 (미지정 시 전체)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="수집 가능한 데이터 목록 출력",
    )
    args = parser.parse_args()

    # --list 옵션: 수집 가능한 데이터 목록 출력 후 종료
    if args.list:
        print("\nAIPA 데이터 수집 소스 목록\n")
        for src_name, cls in COLLECTORS.items():
            collector = cls()
            print(f"  [{src_name}]")
            if hasattr(collector, "TABLES"):
                for k, v in collector.TABLES.items():
                    print(f"    - {k}: {v.get('description', '')}")
            if hasattr(collector, "APIS"):
                for k, v in collector.APIS.items():
                    print(f"    - {k}: {v.get('description', '')}")
            if hasattr(collector, "SEARCH_GROUPS"):
                for g in collector.SEARCH_GROUPS:
                    print(f"    - search: {g['groupName']}")
            if hasattr(collector, "SHOPPING_CATEGORIES"):
                for k in collector.SHOPPING_CATEGORIES:
                    print(f"    - shopping: {k}")
            print()
        return

    # 실제 파이프라인 실행
    sources = [args.source] if args.source else None
    asyncio.run(run_pipeline(sources))  # 비동기 함수를 동기적으로 실행


if __name__ == "__main__":
    main()
