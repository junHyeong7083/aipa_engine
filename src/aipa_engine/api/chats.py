"""
채팅 대화방 저장/조회 API (카톡방식 — 유저별 영구 보관)

모든 라우트가 /chats/{user_id} 로 시작하여 **유저별로 격리**된다.
- 대화 1건 = chat_sessions 1행 (messages JSONB 배열)
- 저장(upsert)/삭제는 공유 Bearer 토큰 필요, 조회는 공개
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..db import get_pool
from .deps import require_bearer

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_pool():
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="데이터베이스를 사용할 수 없습니다")
    return pool


class ChatSaveRequest(BaseModel):
    id: str = Field(min_length=1)              # 세션 id (예: plat_dcinside / persona_교수)
    platform: str | None = None
    title: str = "대화"
    messages: list[dict] = []                  # [{role, content}, ...]


@router.post("/{user_id}", dependencies=[Depends(require_bearer)])
async def save_chat(user_id: str, req: ChatSaveRequest):
    """대화방 저장/업데이트 (upsert). 메시지 올 때마다 호출."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        # 유저 존재 확인 (FK 보호)
        exists = await conn.fetchval("SELECT 1 FROM users WHERE id = $1", user_id)
        if not exists:
            raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다")
        await conn.execute(
            """
            INSERT INTO chat_sessions (id, user_id, platform, title, messages, updated_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
            ON CONFLICT (id) DO UPDATE
            SET title = EXCLUDED.title,
                platform = EXCLUDED.platform,
                messages = EXCLUDED.messages,
                updated_at = NOW()
            """,
            req.id, user_id, req.platform, req.title,
            json.dumps(req.messages, ensure_ascii=False),
        )
    return {"success": True}


@router.get("/{user_id}")
async def list_chats(user_id: str, limit: int = 50):
    """유저의 대화방 목록 (최근순). 마지막 메시지 미리보기 포함."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, platform, title, messages, updated_at
            FROM chat_sessions WHERE user_id = $1
            ORDER BY updated_at DESC LIMIT $2
            """,
            user_id, limit,
        )
    data = []
    for r in rows:
        msgs = json.loads(r["messages"]) if isinstance(r["messages"], str) else (r["messages"] or [])
        last = msgs[-1]["content"] if msgs else ""
        data.append({
            "id": r["id"],
            "platform": r["platform"],
            "title": r["title"],
            "lastMessage": last,
            "messageCount": len(msgs),
            "updatedAt": r["updated_at"].isoformat(),
        })
    return {"success": True, "data": data}


@router.get("/{user_id}/{session_id}")
async def get_chat(user_id: str, session_id: str):
    """대화방 전체 메시지 조회 (재진입용)."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, platform, title, messages FROM chat_sessions WHERE user_id = $1 AND id = $2",
            user_id, session_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
    msgs = json.loads(row["messages"]) if isinstance(row["messages"], str) else (row["messages"] or [])
    return {
        "success": True,
        "data": {
            "id": row["id"],
            "platform": row["platform"],
            "title": row["title"],
            "messages": msgs,
        },
    }


@router.delete("/{user_id}/{session_id}", dependencies=[Depends(require_bearer)])
async def delete_chat(user_id: str, session_id: str):
    """대화방 삭제."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        deleted = await conn.fetchval(
            "DELETE FROM chat_sessions WHERE user_id = $1 AND id = $2 RETURNING id",
            user_id, session_id,
        )
    if not deleted:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
    return {"success": True}
