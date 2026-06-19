"""
공통 API 의존성 (FastAPI Dependencies)

require_bearer: 보호 라우트(PUT/DELETE 등)용 공유 Bearer 토큰 검증.
aishort 백엔드의 middleware/auth.js 와 동일한 방식.
- 환경변수 API_BEARER_TOKEN 이 비어 있으면 503 (인증 미구성).
- 클라이언트는 'Authorization: Bearer <토큰>' 헤더로 동일 값을 보냄.

⚠️ 단기 운용용 공유 토큰 방식. 사용자별 권한 분리가 필요해지면
   per-user JWT 로 교체 (aishort 와 동일한 업그레이드 경로).
"""

import hmac
import logging

from fastapi import Header, HTTPException

from ..config import get_settings

logger = logging.getLogger(__name__)


async def require_bearer(authorization: str = Header(default="")) -> None:
    """Authorization: Bearer <token> 헤더를 공유 토큰과 상수시간 비교."""
    expected = get_settings().api_bearer_token
    if not expected:
        logger.warning("[auth] API_BEARER_TOKEN 미설정 — 보호 라우트 거부")
        raise HTTPException(status_code=503, detail="인증이 구성되지 않았습니다")

    scheme, _, token = authorization.partition(" ")
    # hmac.compare_digest = 타이밍 공격 방지 상수시간 비교
    if scheme != "Bearer" or not token or not hmac.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="인증 실패")
