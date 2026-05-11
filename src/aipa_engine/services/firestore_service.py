"""
Firestore 서비스 (Firebase Cloud Firestore 연동)

C#으로 비유하면 DbContext / Repository 패턴.
파이프라인 데이터, 시뮬레이션 세션 등을 Firestore에 저장/조회.

Firestore 컬렉션 구조:
    pipeline_data/
        {source}_{name}_{date}/     # 파이프라인 수집 데이터
    pipeline_history/
        {auto_id}/                   # 시계열 히스토리
    simulations/
        {session_id}/               # 시뮬레이션 세션
    training_data/
        {auto_id}/                   # 학습 데이터
"""

import logging
import time
from datetime import datetime
from typing import Optional, Any

# firebase_admin = C#의 Firebase NuGet 패키지
import firebase_admin
from firebase_admin import credentials, firestore

from ..config import get_settings

logger = logging.getLogger(__name__)


class FirestoreService:
    """
    Firestore CRUD 서비스 (C#의 public class FirestoreService : IFirestoreService)

    싱글톤: Firebase 앱은 한번만 초기화해야 하므로 클래스 변수로 관리.
    """

    _initialized = False  # Firebase 앱 초기화 여부 (클래스 변수 = C#의 static)
    _db = None            # Firestore 클라이언트 인스턴스

    def __init__(self):
        """
        생성자 - Firebase 앱 초기화 (최초 1회만)

        초기화 방식 2가지:
        1. serviceAccountKey.json 파일이 있으면 → 로컬 개발용 (명시적 인증)
        2. 파일 없으면 → GCP 환경 자동 인증 (Cloud Run 등에서 자동)
        """
        if not FirestoreService._initialized:
            self._init_firebase()

        self.db = FirestoreService._db

    def _init_firebase(self):
        """
        Firebase Admin SDK 초기화

        C#의 FirebaseApp.Create(new AppOptions { ... }) 같은 것.
        """
        try:
            # 이미 초기화된 앱이 있는지 확인
            firebase_admin.get_app()
            logger.info("Firebase already initialized")
        except ValueError:
            # 초기화 안 됐으면 새로 초기화
            try:
                from pathlib import Path
                key_path = Path("serviceAccountKey.json")

                if key_path.exists():
                    # 방법 1: 서비스 계정 키 파일로 인증 (로컬 개발)
                    cred = credentials.Certificate(str(key_path))
                    firebase_admin.initialize_app(cred, {
                        "projectId": "aipa-ceca3",
                    })
                    logger.info("Firebase initialized with serviceAccountKey.json")
                else:
                    # 방법 2: GCP 자동 인증 (Cloud Run, App Engine 등)
                    firebase_admin.initialize_app(options={
                        "projectId": "aipa-ceca3",
                    })
                    logger.info("Firebase initialized with default credentials")

            except Exception as e:
                logger.warning(f"Firebase init failed: {e}. Firestore disabled.")
                FirestoreService._initialized = True
                FirestoreService._db = None
                return

        try:
            FirestoreService._db = firestore.client()
        except Exception as e:
            logger.warning(f"Firestore client creation failed: {e}")
            FirestoreService._db = None
        FirestoreService._initialized = True

    @property
    def available(self) -> bool:
        """Firestore 사용 가능 여부"""
        return self.db is not None

    # ========== 파이프라인 데이터 저장 ==========

    def save_pipeline_data(self, source: str, name: str, data: dict) -> Optional[str]:
        """
        파이프라인 수집 데이터를 Firestore에 저장

        컬렉션: pipeline_data/{source}_{name}_{date}
        + pipeline_history에 시계열 누적

        C#의 DbContext.PipelineData.Add(new PipelineRecord { ... }) 같은 것.
        """
        if not self.available:
            logger.debug("Firestore not available, skipping save")
            return None

        try:
            date_str = datetime.now().strftime("%Y-%m-%d")

            # 1. 최신 데이터 (날짜별 문서 = 덮어쓰기)
            doc_id = f"{source}_{name}_{date_str}"
            doc_ref = self.db.collection("pipeline_data").document(doc_id)
            record = {
                "source": source,
                "name": name,
                "collected_at": datetime.now().isoformat(),
                "date": date_str,
                **data,
            }
            doc_ref.set(record)  # set = 덮어쓰기 (C#의 Upsert)

            # 2. 히스토리 (자동 ID로 추가 = 시계열)
            history_ref = self.db.collection("pipeline_history").document()
            history_ref.set({
                "source": source,
                "name": name,
                "collected_at": datetime.now().isoformat(),
                "date": date_str,
                "data_summary": {
                    k: v for k, v in data.items()
                    if k != "records"  # 원시 records는 용량 문제로 제외
                },
            })

            logger.info(f"  Firestore: saved {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"  Firestore save failed: {e}")
            raise

    def get_pipeline_data(self, source: str, name: str, date: Optional[str] = None) -> Optional[dict]:
        """
        파이프라인 데이터 조회

        date 미지정 시 오늘 데이터 반환.
        C#의 DbContext.PipelineData.FindAsync(id) 같은 것.
        """
        if not self.available:
            return None

        try:
            date_str = date or datetime.now().strftime("%Y-%m-%d")
            doc_id = f"{source}_{name}_{date_str}"
            doc = self.db.collection("pipeline_data").document(doc_id).get()

            if doc.exists:
                return doc.to_dict()
            return None

        except Exception as e:
            logger.warning(f"Firestore read failed: {e}")
            return None

    def get_pipeline_history(self, source: str, name: str, limit: int = 30) -> list[dict]:
        """
        파이프라인 히스토리 조회 (최근 N일)

        C#의 DbContext.PipelineHistory.Where(x => x.Source == source)
                .OrderByDescending(x => x.Date).Take(limit) 같은 것.
        """
        if not self.available:
            return []

        try:
            query = (
                self.db.collection("pipeline_history")
                .where("source", "==", source)
                .where("name", "==", name)
                .order_by("collected_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            docs = query.stream()
            return [doc.to_dict() for doc in docs]

        except Exception as e:
            logger.warning(f"Firestore query failed: {e}")
            return []

    # ========== 시뮬레이션 세션 저장 ==========

    def save_simulation(self, session_id: str, session_data: dict) -> bool:
        """
        시뮬레이션 세션 저장/업데이트

        C#의 DbContext.Simulations.Update(session) 같은 것.
        """
        if not self.available:
            return False

        try:
            doc_ref = self.db.collection("simulations").document(session_id)
            session_data["updated_at"] = datetime.now().isoformat()
            doc_ref.set(session_data, merge=True)  # merge=True = 기존 필드 유지하면서 업데이트
            return True

        except Exception as e:
            logger.error(f"Firestore simulation save failed: {e}")
            raise

    def get_simulation(self, session_id: str) -> Optional[dict]:
        """시뮬레이션 세션 조회"""
        if not self.available:
            return None

        try:
            doc = self.db.collection("simulations").document(session_id).get()
            return doc.to_dict() if doc.exists else None

        except Exception as e:
            logger.warning(f"Firestore simulation read failed: {e}")
            return None

    def list_simulations(self, limit: int = 20) -> list[dict]:
        """최근 시뮬레이션 목록 조회"""
        if not self.available:
            return []

        try:
            query = (
                self.db.collection("simulations")
                .order_by("updated_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            docs = query.stream()
            return [{"id": doc.id, **doc.to_dict()} for doc in docs]

        except Exception as e:
            logger.warning(f"Firestore list failed: {e}")
            return []

    # ========== 학습 데이터 저장 ==========

    def save_training_example(self, example: dict) -> Optional[str]:
        """학습 데이터 1건 저장 (자동 ID)"""
        if not self.available:
            return None

        try:
            doc_ref = self.db.collection("training_data").document()
            example["saved_at"] = datetime.now().isoformat()
            doc_ref.set(example)
            return doc_ref.id

        except Exception as e:
            logger.error(f"Firestore training save failed: {e}")
            raise

    # ========== 배치 쓰기 (Batch Operations) ==========

    # Firestore 배치 제한: 500 operations per batch
    _BATCH_LIMIT = 500

    def batch_write(self, collection: str, documents: list[dict], id_field: Optional[str] = None) -> int:
        """
        여러 문서를 배치로 한번에 저장.

        collection: 저장할 컬렉션 이름
        documents: 저장할 문서 리스트
        id_field: 문서 ID로 사용할 필드명 (없으면 자동 ID 생성)

        반환: 성공적으로 저장된 문서 수.
        Firestore 배치 제한(500)을 자동으로 분할 처리.

        C#의 WriteBatch / BulkWriter 패턴.
        """
        if not self.available:
            logger.warning("Firestore not available, skipping batch write")
            return 0

        total_written = 0
        # 500개씩 분할
        for i in range(0, len(documents), self._BATCH_LIMIT):
            chunk = documents[i:i + self._BATCH_LIMIT]
            batch = self.db.batch()

            for doc_data in chunk:
                if id_field and id_field in doc_data:
                    doc_ref = self.db.collection(collection).document(str(doc_data[id_field]))
                else:
                    doc_ref = self.db.collection(collection).document()
                doc_data["saved_at"] = datetime.now().isoformat()
                batch.set(doc_ref, doc_data)

            try:
                batch.commit()
                total_written += len(chunk)
                logger.info(f"  Firestore batch: wrote {len(chunk)} docs to {collection}")
            except Exception as e:
                logger.error(f"Firestore batch commit failed (chunk {i // self._BATCH_LIMIT + 1}): {e}")
                raise

            # Firestore 쓰기 속도 제한 대응: 배치 간 짧은 대기
            if i + self._BATCH_LIMIT < len(documents):
                time.sleep(0.5)

        return total_written
