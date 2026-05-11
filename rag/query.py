"""
AIPA RAG 검색 모듈
ChromaDB에서 관련 데이터를 검색하여 컨텍스트 생성
"""
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Optional

RAG_DIR = Path(__file__).parent
DB_PATH = str(RAG_DIR / "chroma_db")

DEFAULT_N_RESULTS = 5
DEFAULT_DISTANCE_THRESHOLD = 1.5


class AIPARetriever:
    def __init__(self, n_results: int = DEFAULT_N_RESULTS, distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jhgan/ko-sroberta-multitask"
        )
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.naver_col = self.client.get_collection("naver_trends", embedding_function=self.ef)
        self.kosis_col = self.client.get_collection("kosis_stats", embedding_function=self.ef)
        self.default_n_results = n_results
        self.distance_threshold = distance_threshold

    def _filter_by_distance(self, results: dict) -> dict:
        """거리 임계값(distance_threshold)을 초과하는 결과를 제거"""
        if not results.get("distances") or not results["distances"][0]:
            return results

        filtered_indices = [
            i for i, d in enumerate(results["distances"][0])
            if d <= self.distance_threshold
        ]

        filtered = {}
        for key in results:
            if results[key] is None:
                filtered[key] = None
            elif isinstance(results[key], list) and len(results[key]) > 0 and isinstance(results[key][0], list):
                filtered[key] = [[results[key][0][i] for i in filtered_indices]]
            else:
                filtered[key] = results[key]
        return filtered

    def _build_where_filter(self, date: Optional[str] = None, source_type: Optional[str] = None) -> Optional[dict]:
        """메타데이터 필터 조건 생성"""
        conditions = []
        if date:
            conditions.append({"date": date})
        if source_type:
            conditions.append({"type": source_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def search(self, query: str, n_results: Optional[int] = None,
               date: Optional[str] = None, source_type: Optional[str] = None) -> dict:
        """트렌드 + 통계 모두 검색"""
        n = n_results or self.default_n_results
        where_filter = self._build_where_filter(date, source_type)
        kwargs = {"query_texts": [query], "n_results": n}
        if where_filter:
            kwargs["where"] = where_filter

        trends = self.naver_col.query(**kwargs)
        stats = self.kosis_col.query(**kwargs)
        return {
            "trends": self._filter_by_distance(trends),
            "stats": self._filter_by_distance(stats),
        }

    def search_trends(self, query: str, n_results: Optional[int] = None,
                      date: Optional[str] = None, source_type: Optional[str] = None) -> dict:
        """네이버 트렌드만 검색"""
        n = n_results or self.default_n_results
        where_filter = self._build_where_filter(date, source_type)
        kwargs = {"query_texts": [query], "n_results": n}
        if where_filter:
            kwargs["where"] = where_filter
        return self._filter_by_distance(self.naver_col.query(**kwargs))

    def search_stats(self, query: str, n_results: Optional[int] = None,
                     date: Optional[str] = None, source_type: Optional[str] = None) -> dict:
        """통계 데이터만 검색"""
        n = n_results or self.default_n_results
        where_filter = self._build_where_filter(date, source_type)
        kwargs = {"query_texts": [query], "n_results": n}
        if where_filter:
            kwargs["where"] = where_filter
        return self._filter_by_distance(self.kosis_col.query(**kwargs))

    def build_context(self, query: str, n_results: Optional[int] = None,
                      date: Optional[str] = None, source_type: Optional[str] = None) -> str:
        """검색 결과를 프롬프트용 컨텍스트 문자열로 변환"""
        n = n_results or 3
        results = self.search(query, n_results=n, date=date, source_type=source_type)
        context_parts = []

        # 트렌드 컨텍스트
        if results["trends"]["documents"][0]:
            context_parts.append("[최근 시장 트렌드]")
            for doc in results["trends"]["documents"][0]:
                context_parts.append(f"- {doc}")

        # 통계 컨텍스트
        if results["stats"]["documents"][0]:
            context_parts.append("\n[관련 통계 데이터]")
            for doc in results["stats"]["documents"][0]:
                context_parts.append(f"- {doc}")

        return "\n".join(context_parts)
