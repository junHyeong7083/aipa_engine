"""
AIPA Engine - FastAPI 애플리케이션 진입점 (Entry Point)

C#으로 비유하면 Program.cs + Startup.cs 역할.
FastAPI = C#의 ASP.NET Core와 동일한 웹 프레임워크.
"""

import logging
import time

# asynccontextmanager = C#의 IHostedService.StartAsync/StopAsync 같은 라이프사이클 관리
from contextlib import asynccontextmanager
# FastAPI = C#의 WebApplication.CreateBuilder() 같은 것
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# CORS 미들웨어 = C#의 app.UseCors() 와 동일
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# 우리가 만든 설정 모듈에서 설정 가져오기
from .config import get_settings
from .logging_config import setup_logging, request_id_var
# Request ID 추적 & Rate Limiting 미들웨어
from .middleware import RequestIDMiddleware, RateLimitMiddleware

logger = logging.getLogger("aipa.main")
# API 라우터 가져오기 (C#의 MapControllers() 같은 것)
from .api import router as api_router


# C#의 IHostedService 같은 역할 - 앱 시작/종료 시 실행되는 코드
# @asynccontextmanager = 비동기 리소스 관리 (C#의 using 문과 비슷한 개념)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 이벤트 (C#의 Program.cs에서 app.Run() 전후 코드와 동일)"""
    # ===== 시작 시 실행 (Startup) =====
    settings = get_settings()                              # 설정 로드 (triggers config validation)
    setup_logging(debug=settings.debug)                    # JSON 구조화 로깅 초기화
    logger.info(f"Starting {settings.app_name} v0.1.0")
    logger.info(f"Debug mode: {settings.debug}")
    if not settings.anthropic_api_key:
        logger.warning("Anthropic API key not configured - LLM features will use mock responses")

    yield  # ← 여기서 앱이 실행됨 (C#의 app.Run()에 해당)

    # ===== 종료 시 실행 (Shutdown) =====
    print("Shutting down AIPA Engine")


# C#의 WebApplication.CreateBuilder() + builder.Build() 합친 것
def create_app() -> FastAPI:
    """FastAPI 앱 생성 및 설정 (C#의 Startup.ConfigureServices + Configure 합친 것)"""
    settings = get_settings()

    # FastAPI 앱 인스턴스 생성 (C#의 var app = builder.Build())
    app = FastAPI(
        title=settings.app_name,                                    # Swagger UI에 표시될 제목
        description="Statistical Persona Survey Simulation Engine",  # 설명
        version="0.1.0",
        lifespan=lifespan,                                          # 시작/종료 이벤트 연결
    )

    # --- Global exception handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url.path, exc,
            exc_info=True,
            extra={"http_method": request.method, "http_path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # --- Request/Response logging middleware (Pure ASGI) ---
    from .api.monitoring import metrics as _metrics_collector

    class LoggingMiddleware:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return

            start = time.time()
            status_code = 200

            async def send_with_logging(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            await self.app(scope, receive, send_with_logging)

            duration_ms = round((time.time() - start) * 1000, 1)
            method = scope.get("method", "")
            path = scope.get("path", "")
            logger.info(
                "%s %s -> %s (%.0fms)", method, path, status_code, duration_ms,
                extra={"http_method": method, "http_path": path, "http_status": status_code, "duration_ms": duration_ms},
            )
            _metrics_collector.record_request(path, status_code, duration_ms)

    app.add_middleware(LoggingMiddleware)

    # CORS 미들웨어 추가 (C#의 app.UseCors(policy => policy.AllowAnyOrigin()) 과 동일)
    # 다른 도메인(예: Flutter 웹앱)에서 이 API를 호출할 수 있게 허용
    # 기본값은 localhost만 허용. 프로덕션에서는 ALLOWED_ORIGINS 환경변수로 도메인 지정 필요.
    default_origins = ["http://localhost:3000", "http://localhost:8080", "http://localhost:5000"]
    allowed_origins = settings.allowed_origins.split(",") if settings.allowed_origins else default_origins
    if "*" in allowed_origins:
        logger.warning("CORS: allow_origins=['*'] is insecure. Set ALLOWED_ORIGINS in production.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,      # 허용할 도메인 목록
        allow_credentials=True,             # 쿠키/인증 헤더 허용
        allow_methods=["*"],                # 모든 HTTP 메서드 허용 (GET, POST 등)
        allow_headers=["*"],                # 모든 헤더 허용
    )

    # Rate Limiting 미들웨어 (IP 기반 인메모리 슬라이딩 윈도우)
    # CORS 안쪽, RequestID 안쪽에서 동작
    app.add_middleware(RateLimitMiddleware)

    # Request ID 추적 미들웨어 (가장 바깥쪽 - 모든 요청에 X-Request-ID 부여)
    # Starlette는 add_middleware 순서가 LIFO이므로 마지막에 추가한 것이 outermost
    app.add_middleware(RequestIDMiddleware)

    # API 라우터 등록 (C#의 app.MapControllers() 같은 것)
    # prefix="/api/v1" → 모든 API 경로 앞에 /api/v1 이 붙음
    app.include_router(api_router, prefix=settings.api_prefix)

    # 헬스체크 엔드포인트 (C#의 app.MapGet("/health", ...) 과 동일)
    # 서버가 살아있는지 확인하는 용도 (GCP Cloud Run, 로드밸런서 등이 사용)
    @app.get("/health")
    async def health_check():
        """헬스체크 - 서버 상태 확인용"""
        return {"status": "healthy", "version": "0.1.0"}

    return app


# 앱 인스턴스 생성 (C#의 var app = builder.Build() 후 전역 변수로 저장)
app = create_app()


# 직접 실행 시 (python -m aipa_engine.main)
# C#의 dotnet run 같은 것
if __name__ == "__main__":
    import os
    import uvicorn  # uvicorn = C#의 Kestrel 웹서버와 동일한 역할

    port = int(os.environ.get("PORT", 8080))  # 포트 번호 (기본 8080)
    # reload=True → 코드 변경 시 자동 재시작 (C#의 dotnet watch run 같은 것)
    uvicorn.run("aipa_engine.main:app", host="0.0.0.0", port=port, reload=True)
