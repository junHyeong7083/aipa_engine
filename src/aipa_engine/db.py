"""
PostgreSQL 연결 + 스키마 부트스트랩 (Firestore 대체)

aishort 백엔드(Node + pg)와 동일한 컨벤션을 Python/asyncpg 로 옮긴 것.
- 앱 부팅 시 커넥션 풀 생성 + 테이블 자동 생성 (index.js 의 init 패턴)
- 이벤트 루프를 막지 않도록 동기 드라이버(psycopg) 대신 비동기 asyncpg 사용

테이블:
    users            - 사용자 계정 (local/kakao/google)
    survey_history   - 사용자별 설문 히스토리
    simulations      - 시뮬레이션 세션 (완료분 저장)
    pipeline_data    - 파이프라인 수집 데이터 (날짜별 최신)
    pipeline_history - 파이프라인 시계열 히스토리
    training_data    - 학습 데이터
"""

import logging
from typing import Optional

import asyncpg

from .config import get_settings

logger = logging.getLogger(__name__)

# 전역 커넥션 풀 (C#의 싱글톤 DbContextPool 같은 것)
_pool: Optional[asyncpg.Pool] = None


# 스키마 DDL — 부팅 시 IF NOT EXISTS 로 멱등 생성
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY,                       -- 클라이언트 생성 id (kakao/google uid 등)
    email               TEXT UNIQUE NOT NULL,
    password            TEXT NOT NULL DEFAULT '',               -- bcrypt 해시 (OAuth 계정은 빈 문자열)
    nickname            TEXT NOT NULL DEFAULT '사용자',
    auth_method         TEXT NOT NULL DEFAULT 'local',          -- local | kakao | google
    profile_image_url   TEXT NOT NULL DEFAULT '',
    interests           JSONB NOT NULL DEFAULT '[]'::jsonb,
    plan                TEXT NOT NULL DEFAULT 'free',           -- free | plus | pro
    plan_expires_at     TIMESTAMPTZ,
    surveys_remaining   INTEGER NOT NULL DEFAULT 3,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS survey_history (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT NOT NULL DEFAULT '설문 시뮬레이션',
    persona_count   INTEGER NOT NULL DEFAULT 0,
    accuracy        REAL NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'completed',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_survey_history_user ON survey_history(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS simulations (
    session_id  TEXT PRIMARY KEY,
    data        JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_simulations_updated ON simulations(updated_at DESC);

CREATE TABLE IF NOT EXISTS pipeline_data (
    doc_id          TEXT PRIMARY KEY,                           -- {source}_{name}_{date}
    source          TEXT NOT NULL,
    name            TEXT NOT NULL,
    date            TEXT NOT NULL,
    data            JSONB NOT NULL,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_history (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    name            TEXT NOT NULL,
    date            TEXT NOT NULL,
    data_summary    JSONB NOT NULL DEFAULT '{}'::jsonb,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pipeline_history_key ON pipeline_history(source, name, collected_at DESC);

CREATE TABLE IF NOT EXISTS training_data (
    id          BIGSERIAL PRIMARY KEY,
    data        JSONB NOT NULL,
    saved_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def init_db() -> Optional[asyncpg.Pool]:
    """
    커넥션 풀 생성 + 스키마 부트스트랩.

    DB 접속 실패 시 None 을 반환하고 앱은 계속 구동 (Firestore 가 그랬듯 graceful degrade).
    호출처: main.py 의 lifespan.
    """
    global _pool
    if _pool is not None:
        return _pool

    settings = get_settings()
    if not settings.db_password:
        logger.warning("[db] DB_PASSWORD 미설정 — PostgreSQL 비활성화 (데이터 저장 기능 off)")
        return None

    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
            command_timeout=30,
        )
        async with _pool.acquire() as conn:
            await conn.execute(_SCHEMA_DDL)
        logger.info("[db] PostgreSQL 연결 + 스키마 준비 완료")
        return _pool
    except Exception as e:
        logger.warning(f"[db] PostgreSQL 연결 실패: {e}. 데이터 저장 비활성화.")
        _pool = None
        return None


async def close_db() -> None:
    """앱 종료 시 풀 정리 (lifespan shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("[db] PostgreSQL 풀 종료")


def get_pool() -> Optional[asyncpg.Pool]:
    """현재 커넥션 풀 반환 (없으면 None)."""
    return _pool
