"""
설정 관리 모듈 (Application Configuration)

C#으로 비유하면 appsettings.json + IConfiguration 패턴과 동일.
.env 파일에서 환경변수를 읽어서 Settings 객체로 관리함.
"""

# lru_cache = C#의 Lazy<T> 패턴과 비슷. 한번 만든 객체를 캐싱해서 재사용
import logging
import warnings
from functools import lru_cache
# BaseSettings = C#의 IOptions<T> 패턴. .env 파일 → 자동으로 속성에 매핑됨
from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator


# C#의 AppSettings 클래스와 동일한 역할
# .env 파일의 키 이름과 변수명이 자동 매칭됨 (대소문자 무시)
# 예: .env의 KOSIS_API_KEY → self.kosis_api_key
class Settings(BaseSettings):
    """환경변수에서 로드되는 앱 설정 (C#의 appsettings.json 역할)"""

    # --- API 기본 설정 ---
    app_name: str = "AIPA Engine"           # 앱 이름
    debug: bool = False                      # 디버그 모드 (C#의 ASPNETCORE_ENVIRONMENT와 비슷)
    api_prefix: str = "/api/v1"             # API URL 접두사 (C#의 [Route("api/v1")] 같은 것)

    # --- KOSIS API (통계청 공공데이터) ---
    kosis_api_key: str = ""                 # 통계청 API 키

    # --- Anthropic API (Claude AI) ---
    anthropic_api_key: str = ""             # Claude API 키
    anthropic_model: str = "claude-sonnet-4-20250514"  # 사용할 Claude 모델명

    # --- 네이버 데이터랩 API ---
    naver_client_id: str = ""               # 네이버 개발자센터 Client ID
    naver_client_secret: str = ""           # 네이버 개발자센터 Client Secret

    # --- CORS 설정 (C#의 services.AddCors()와 동일) ---
    allowed_origins: str = ""  # 쉼표로 구분된 허용 도메인. 비어있으면 모든 도메인 허용(*)

    # --- 시뮬레이션 기본값 ---
    default_panel_count: int = 10           # 기본 패널(응답자) 수
    max_panel_count: int = 200              # 최대 패널 수

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 60                # 일반 엔드포인트 분당 최대 요청 수
    simulation_rate_limit_per_minute: int = 10     # POST /simulations 분당 최대 요청 수

    @field_validator("kosis_api_key", "anthropic_api_key", "naver_client_id", "naver_client_secret", mode="after")
    @classmethod
    def _check_not_placeholder(cls, v: str, info) -> str:
        placeholders = {"", "your-api-key-here", "your_api_key", "changeme", "xxx", "test"}
        if v.strip().lower() in placeholders:
            return ""  # treat placeholders as empty
        return v

    @model_validator(mode="after")
    def _warn_missing_keys(self):
        _logger = logging.getLogger("aipa.config")
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.kosis_api_key:
            missing.append("KOSIS_API_KEY")
        if not self.naver_client_id or not self.naver_client_secret:
            missing.append("NAVER_CLIENT_ID/SECRET")
        if missing:
            _logger.warning(f"[config] Missing API keys at startup: {', '.join(missing)}. Related features will be disabled.")
        return self

    # C#의 ConfigurationBuilder().AddJsonFile("appsettings.json") 같은 역할
    # .env 파일 경로와 인코딩을 지정
    class Config:
        env_file = ".env"                   # 환경변수 파일 경로
        env_file_encoding = "utf-8"         # 파일 인코딩


# C#의 services.AddSingleton<Settings>() 같은 역할
# @lru_cache 덕분에 처음 호출 시 1번만 생성되고, 이후엔 같은 인스턴스 반환
# → 매번 .env 파일을 다시 읽지 않음
@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤 인스턴스 반환 (C#의 DI에서 IOptions<Settings> 주입받는 것과 동일)"""
    return Settings()
