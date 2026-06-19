"""
[DEPRECATED] firestore_service → db_service 로 이전됨.

Firestore(firebase-admin) 의존성은 제거되었습니다. PostgreSQL 기반
PostgresService 로 대체되었으며, 기존 import 경로 호환을 위해 여기서 재export 합니다.

새 코드는 `from ..services.db_service import PostgresService` 를 사용하세요.
"""

from .db_service import PostgresService, FirestoreService

__all__ = ["PostgresService", "FirestoreService"]
