"""
Structured JSON Logging Configuration

GCP Cloud Logging 등 로그 수집 시스템과 호환되는 JSON 형식 로그 출력.
C#으로 비유하면 Serilog + JsonFormatter 설정과 동일한 역할.

사용법:
    from .logging_config import setup_logging
    setup_logging(debug=True)
"""

import json
import logging
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone

# request_id를 요청 단위로 추적하기 위한 context variable
# C#의 AsyncLocal<string> 또는 HttpContext.TraceIdentifier 와 동일한 역할
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """
    각 로그 라인을 JSON 객체로 출력하는 포맷터.

    C#의 Serilog JsonFormatter 와 동일한 역할.
    Cloud Logging, ELK, Datadog 등에서 자동 파싱 가능.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # request_id 추가 (contextvars 에서 가져옴, 없으면 생략)
        rid = request_id_var.get(None)
        if rid is not None:
            log_entry["request_id"] = rid

        # extra 필드: 사용자가 logger.info("msg", extra={"key": "val"}) 로 전달한 값
        # logging 모듈이 자동으로 추가하는 내부 속성은 제외
        _BUILTIN_ATTRS = frozenset({
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "pathname", "filename", "module", "levelno", "levelname",
            "thread", "threadName", "process", "processName",
            "getMessage", "message", "msecs", "taskName",
        })
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in _BUILTIN_ATTRS and not k.startswith("_")
        }
        if extra:
            log_entry["extra"] = extra

        # 예외 정보가 있으면 structured 형태로 추가
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            log_entry["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": traceback.format_exception(exc_type, exc_value, exc_tb),
            }

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logging(debug: bool = False) -> None:
    """
    루트 로거에 JSON 포맷 핸들러를 설정한다.

    C#으로 비유하면 Host.CreateDefaultBuilder() 에서
    .ConfigureLogging(builder => builder.AddJsonConsole()) 하는 것과 동일.

    Args:
        debug: True이면 DEBUG 레벨, False이면 INFO 레벨로 설정
    """
    level = logging.DEBUG if debug else logging.INFO

    # 루트 로거 설정
    root = logging.getLogger()
    root.setLevel(level)

    # 기존 핸들러 제거 (중복 방지)
    root.handlers.clear()

    # JSON 포맷 StreamHandler 추가 (stdout 으로 출력)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # 서드파티 로거 노이즈 억제
    # C#에서 appsettings.json 의 "Logging": {"LogLevel": {"Microsoft": "Warning"}} 과 동일
    # uvicorn.access/error는 억제하지 않음 (startup complete, shutdown 메시지 필요)
    noisy_loggers = [
        "httpx",
        "httpcore",
        "hpack",
        "watchfiles",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("aipa").debug("JSON structured logging initialised (level=%s)", level)
