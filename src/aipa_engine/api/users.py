"""
사용자 + 설문 히스토리 API (Firestore users 컬렉션 대체)

aishort 백엔드 routes/users.js 를 Python/FastAPI/asyncpg 로 옮긴 것.
- 회원가입/로그인은 공개, 수정/삭제는 공유 Bearer 토큰 필요.
- OAuth(kakao/google) 가입은 비밀번호 없이 클라이언트가 만든 id 로 등록.
- 응답 형태: {"success": bool, "data"|"error"|"message": ...} (aishort 와 동일).
"""

import json
import logging

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..db import get_pool
from .deps import require_bearer

logger = logging.getLogger(__name__)

router = APIRouter()

# PUT 시 사용자가 변경 가능한 컬럼만 허용 (임의 컬럼 덮어쓰기 방지)
_PUT_ALLOWED = {
    "nickname",
    "profile_image_url",
    "interests",
    "plan",
    "plan_expires_at",
    "surveys_remaining",
    "last_login_at",
}

# GET/응답에서 노출할 컬럼 (password 제외)
_PUBLIC_COLS = (
    "id, email, nickname, auth_method, profile_image_url, interests, "
    "plan, plan_expires_at, surveys_remaining, created_at, last_login_at"
)


def _row_to_user(row) -> dict:
    """asyncpg Record → JSON 직렬화 가능한 dict (interests JSONB 파싱, 날짜 ISO)."""
    d = dict(row)
    if isinstance(d.get("interests"), str):
        d["interests"] = json.loads(d["interests"])
    for k in ("created_at", "last_login_at", "plan_expires_at"):
        if d.get(k) is not None and hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    return d


def _require_pool():
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="데이터베이스를 사용할 수 없습니다")
    return pool


# ===== 요청 DTO =====

class SignupRequest(BaseModel):
    id: str = Field(min_length=1)                 # 클라이언트 생성 id (kakao/google uid 등)
    email: str
    password: str | None = None                   # local 가입 시 필수
    nickname: str | None = None
    auth_method: str = "local"                    # local | kakao | google
    profile_image_url: str | None = None
    interests: list[str] = []


class LoginRequest(BaseModel):
    email: str
    password: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6)


# ===== 엔드포인트 =====

@router.post("/")
async def signup(req: SignupRequest):
    """회원가입 (공개). local 은 비밀번호 필수, OAuth 는 비밀번호 없음."""
    pool = _require_pool()

    method = req.auth_method or "local"
    password_hash = ""
    if method == "local":
        if not req.password or len(req.password) < 6:
            raise HTTPException(status_code=400, detail="비밀번호는 최소 6자 이상이어야 합니다")
        password_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()

    async with pool.acquire() as conn:
        existing = await conn.fetchval("SELECT id FROM users WHERE email = $1", req.email)
        if existing:
            raise HTTPException(status_code=409, detail="이미 사용 중인 이메일입니다")
        try:
            row = await conn.fetchrow(
                f"""
                INSERT INTO users (id, email, password, nickname, auth_method, profile_image_url, interests)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                RETURNING {_PUBLIC_COLS}
                """,
                req.id, req.email, password_hash, req.nickname or "사용자",
                method, req.profile_image_url or "", json.dumps(req.interests, ensure_ascii=False),
            )
        except Exception as e:
            # id 중복 등
            logger.error(f"signup failed: {e}")
            raise HTTPException(status_code=409, detail="이미 존재하는 사용자입니다")

    return {"success": True, "data": _row_to_user(row)}


@router.post("/login")
async def login(req: LoginRequest):
    """로그인 (공개, local 한정). OAuth 계정은 클라이언트가 GET /users/{id} 로 조회."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE email = $1 AND auth_method = 'local'", req.email
        )
        if not row:
            raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다")
        if not bcrypt.checkpw(req.password.encode(), (row["password"] or "").encode()):
            raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다")
        await conn.execute("UPDATE users SET last_login_at = NOW() WHERE id = $1", row["id"])
        fresh = await conn.fetchrow(f"SELECT {_PUBLIC_COLS} FROM users WHERE id = $1", row["id"])

    return {"success": True, "data": _row_to_user(fresh)}


@router.get("/{user_id}")
async def get_user(user_id: str):
    """유저 정보 조회 (공개, password 제외)."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT {_PUBLIC_COLS} FROM users WHERE id = $1", user_id)
    if not row:
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다")
    return {"success": True, "data": _row_to_user(row)}


@router.put("/{user_id}", dependencies=[Depends(require_bearer)])
async def update_user(user_id: str, fields: dict):
    """유저 정보 수정 (Bearer 필요, 허용 컬럼만)."""
    pool = _require_pool()
    sets, values, idx = [], [], 1
    for key, value in fields.items():
        if key not in _PUT_ALLOWED:
            continue
        if key == "interests":
            sets.append(f"interests = ${idx}::jsonb")
            values.append(json.dumps(value, ensure_ascii=False))
        else:
            sets.append(f"{key} = ${idx}")
            values.append(value)
        idx += 1

    if not sets:
        raise HTTPException(status_code=400, detail="변경 가능한 필드가 없습니다")

    values.append(user_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"UPDATE users SET {', '.join(sets)} WHERE id = ${idx} RETURNING {_PUBLIC_COLS}",
            *values,
        )
    if not row:
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다")
    return {"success": True, "data": _row_to_user(row)}


@router.put("/{user_id}/password", dependencies=[Depends(require_bearer)])
async def change_password(user_id: str, req: PasswordChangeRequest):
    """비밀번호 변경 (Bearer + 현재 비밀번호 검증, local 계정만)."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT password, auth_method FROM users WHERE id = $1", user_id)
        if not row:
            raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다")
        if row["auth_method"] != "local":
            raise HTTPException(status_code=400, detail="OAuth 계정은 비밀번호 변경 불가")
        if not bcrypt.checkpw(req.current_password.encode(), (row["password"] or "").encode()):
            raise HTTPException(status_code=401, detail="현재 비밀번호 불일치")
        new_hash = bcrypt.hashpw(req.new_password.encode(), bcrypt.gensalt()).decode()
        await conn.execute("UPDATE users SET password = $1 WHERE id = $2", new_hash, user_id)
    return {"success": True, "message": "비밀번호 변경 완료"}


@router.delete("/{user_id}", dependencies=[Depends(require_bearer)])
async def delete_user(user_id: str):
    """회원 탈퇴 (Bearer 필요). 히스토리는 FK ON DELETE CASCADE 로 함께 삭제."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        deleted = await conn.fetchval("DELETE FROM users WHERE id = $1 RETURNING id", user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="유저를 찾을 수 없습니다")
    return {"success": True, "message": "회원 탈퇴 완료"}


# ===== 설문 히스토리 =====

class SurveyHistoryRequest(BaseModel):
    id: str
    title: str = "설문 시뮬레이션"
    persona_count: int = 0
    accuracy: float = 0.0
    status: str = "completed"


@router.post("/{user_id}/history", dependencies=[Depends(require_bearer)])
async def save_history(user_id: str, req: SurveyHistoryRequest):
    """설문 히스토리 저장 (Bearer 필요)."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO survey_history (id, user_id, title, persona_count, accuracy, status)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO UPDATE
            SET title = EXCLUDED.title, persona_count = EXCLUDED.persona_count,
                accuracy = EXCLUDED.accuracy, status = EXCLUDED.status
            """,
            req.id, user_id, req.title, req.persona_count, req.accuracy, req.status,
        )
    return {"success": True, "message": "히스토리 저장 완료"}


@router.get("/{user_id}/history")
async def list_history(user_id: str, limit: int = 50):
    """설문 히스토리 조회 (최근순)."""
    pool = _require_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, persona_count, accuracy, status, created_at
            FROM survey_history WHERE user_id = $1
            ORDER BY created_at DESC LIMIT $2
            """,
            user_id, limit,
        )
    data = [
        {
            "id": r["id"],
            "title": r["title"],
            "personaCount": r["persona_count"],
            "accuracy": r["accuracy"],
            "status": r["status"],
            "createdAt": r["created_at"].isoformat(),
        }
        for r in rows
    ]
    return {"success": True, "data": data}
