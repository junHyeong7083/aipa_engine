"""
PostgreSQL 데이터 서비스 (기존 FirestoreService 대체)

기존 firestore_service.FirestoreService 와 동일한 메서드 시그니처를 유지하되,
- 동기 → 비동기(async) 로 전환 (FastAPI 이벤트 루프 블로킹 방지)
- Firestore 컬렉션/문서 → PostgreSQL 테이블/JSONB 로 매핑

호출처(simulations.py, monitoring.py)는 메서드를 await 로 호출하도록 변경됨.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from ..db import get_pool

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PostgresService:
    """
    PostgreSQL CRUD 서비스 (기존 FirestoreService 와 동일 역할).

    커넥션 풀은 db.init_db() 가 앱 부팅 시 생성. 여기서는 get_pool() 로 가져다 씀.
    풀이 없으면(available=False) 모든 쓰기는 조용히 skip — Firestore 비활성화 동작과 동일.
    """

    @property
    def available(self) -> bool:
        """DB 사용 가능 여부 (풀이 생성돼 있는지)."""
        return get_pool() is not None

    # ========== 파이프라인 데이터 ==========

    async def save_pipeline_data(self, source: str, name: str, data: dict) -> Optional[str]:
        """파이프라인 수집 데이터 저장 (날짜별 upsert) + 히스토리 누적."""
        pool = get_pool()
        if pool is None:
            logger.debug("DB not available, skipping save")
            return None

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        doc_id = f"{source}_{name}_{date_str}"
        # records 는 용량 문제로 히스토리 요약에서 제외 (Firestore 와 동일)
        summary = {k: v for k, v in data.items() if k != "records"}

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        """
                        INSERT INTO pipeline_data (doc_id, source, name, date, data, collected_at)
                        VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
                        ON CONFLICT (doc_id) DO UPDATE
                        SET data = EXCLUDED.data, collected_at = NOW()
                        """,
                        doc_id, source, name, date_str, json.dumps(data, ensure_ascii=False),
                    )
                    await conn.execute(
                        """
                        INSERT INTO pipeline_history (source, name, date, data_summary)
                        VALUES ($1, $2, $3, $4::jsonb)
                        """,
                        source, name, date_str, json.dumps(summary, ensure_ascii=False),
                    )
            logger.info(f"  DB: saved {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"  DB save failed: {e}")
            raise

    async def get_pipeline_data(
        self, source: str, name: str, date: Optional[str] = None
    ) -> Optional[dict]:
        """파이프라인 데이터 조회 (date 미지정 시 오늘)."""
        pool = get_pool()
        if pool is None:
            return None

        date_str = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        doc_id = f"{source}_{name}_{date_str}"
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT data FROM pipeline_data WHERE doc_id = $1", doc_id
                )
            return json.loads(row["data"]) if row else None
        except Exception as e:
            logger.warning(f"DB read failed: {e}")
            return None

    async def get_pipeline_history(
        self, source: str, name: str, limit: int = 30
    ) -> list[dict]:
        """파이프라인 히스토리 조회 (최근 N건)."""
        pool = get_pool()
        if pool is None:
            return []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT source, name, date, data_summary, collected_at
                    FROM pipeline_history
                    WHERE source = $1 AND name = $2
                    ORDER BY collected_at DESC
                    LIMIT $3
                    """,
                    source, name, limit,
                )
            return [
                {
                    "source": r["source"],
                    "name": r["name"],
                    "date": r["date"],
                    "collected_at": r["collected_at"].isoformat(),
                    "data_summary": json.loads(r["data_summary"]),
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"DB query failed: {e}")
            return []

    # ========== 시뮬레이션 세션 ==========

    async def save_simulation(self, session_id: str, session_data: dict) -> bool:
        """시뮬레이션 세션 저장/업데이트 (upsert)."""
        pool = get_pool()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO simulations (session_id, data, updated_at)
                    VALUES ($1, $2::jsonb, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                    SET data = EXCLUDED.data, updated_at = NOW()
                    """,
                    session_id, json.dumps(session_data, ensure_ascii=False),
                )
            return True
        except Exception as e:
            logger.error(f"DB simulation save failed: {e}")
            raise

    async def get_simulation(self, session_id: str) -> Optional[dict]:
        """시뮬레이션 세션 조회."""
        pool = get_pool()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT data FROM simulations WHERE session_id = $1", session_id
                )
            return json.loads(row["data"]) if row else None
        except Exception as e:
            logger.warning(f"DB simulation read failed: {e}")
            return None

    async def list_simulations(self, limit: int = 20) -> list[dict]:
        """최근 시뮬레이션 목록."""
        pool = get_pool()
        if pool is None:
            return []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT session_id, data FROM simulations ORDER BY updated_at DESC LIMIT $1",
                    limit,
                )
            return [{"id": r["session_id"], **json.loads(r["data"])} for r in rows]
        except Exception as e:
            logger.warning(f"DB list failed: {e}")
            return []

    # ========== 학습 데이터 ==========

    async def save_training_example(self, example: dict) -> Optional[str]:
        """학습 데이터 1건 저장."""
        pool = get_pool()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                new_id = await conn.fetchval(
                    "INSERT INTO training_data (data) VALUES ($1::jsonb) RETURNING id",
                    json.dumps(example, ensure_ascii=False),
                )
            return str(new_id)
        except Exception as e:
            logger.error(f"DB training save failed: {e}")
            raise

    async def batch_write(
        self, collection: str, documents: list[dict], id_field: Optional[str] = None
    ) -> int:
        """
        여러 문서를 한 트랜잭션으로 저장.

        collection: training_data | simulations | pipeline_data 만 지원.
        id_field: 문서 ID 컬럼으로 쓸 필드명 (training_data 는 무시).
        반환: 저장된 행 수.
        """
        pool = get_pool()
        if pool is None:
            logger.warning("DB not available, skipping batch write")
            return 0
        if not documents:
            return 0

        saved_at = _now_iso()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    if collection == "training_data":
                        await conn.executemany(
                            "INSERT INTO training_data (data) VALUES ($1::jsonb)",
                            [(json.dumps({**d, "saved_at": saved_at}, ensure_ascii=False),) for d in documents],
                        )
                    elif collection == "simulations":
                        await conn.executemany(
                            """
                            INSERT INTO simulations (session_id, data, updated_at)
                            VALUES ($1, $2::jsonb, NOW())
                            ON CONFLICT (session_id) DO UPDATE
                            SET data = EXCLUDED.data, updated_at = NOW()
                            """,
                            [
                                (str(d.get(id_field) if id_field else d.get("session_id", "")),
                                 json.dumps(d, ensure_ascii=False))
                                for d in documents
                            ],
                        )
                    else:
                        raise ValueError(f"batch_write: 지원하지 않는 collection '{collection}'")
            logger.info(f"  DB batch: wrote {len(documents)} rows to {collection}")
            return len(documents)
        except Exception as e:
            logger.error(f"DB batch write failed: {e}")
            raise


# 기존 import 경로 호환: `from ..services.firestore_service import FirestoreService` 를
# 한 번에 바꾸기 어려운 곳을 위해 별칭 제공.
FirestoreService = PostgresService
