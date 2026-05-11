"""
AIPA RAG 인덱스 구축
수집된 네이버/KOSIS 데이터를 ChromaDB에 임베딩

Usage:
  python build_index.py                  # 전체 재구축 (full rebuild)
  python build_index.py --incremental    # 마지막 인덱싱 이후 변경된 파일만 처리
  python build_index.py --incremental --dry-run  # 변경 파일 확인만 (인덱싱 안 함)
"""
import argparse
import json
import glob
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

RAG_DIR = Path(__file__).parent
DATA_DIR = RAG_DIR.parent / "data" / "pipeline" / "daily"
DB_PATH = str(RAG_DIR / "chroma_db")
METADATA_PATH = RAG_DIR / "index_metadata.json"


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """Load index metadata from disk, or return empty defaults."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_metadata(meta: dict) -> None:
    """Persist index metadata to disk."""
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File-filtering helpers
# ---------------------------------------------------------------------------

def _get_json_files(pattern: str) -> list[str]:
    """Return sorted JSON file paths matching *pattern*, excluding history files."""
    return [
        p for p in sorted(glob.glob(pattern))
        if "history" not in p
    ]


def _filter_modified_since(paths: list[str], since_ts: float) -> list[str]:
    """Return only paths whose mtime is strictly after *since_ts*."""
    return [p for p in paths if os.path.getmtime(p) > since_ts]


# ---------------------------------------------------------------------------
# Document builders  (extracted from original inline code)
# ---------------------------------------------------------------------------

def _build_naver_search_docs(fpath: str):
    """Yield (doc, doc_id, meta) tuples for naver search-trend file."""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [경고] 파일 읽기 실패 (건너뜀): {fpath} - {e}")
        return

    for result in data.get("results", []):
        title = result.get("title", "")
        keywords = result.get("keywords", [])
        periods = result.get("data", [])
        if not periods:
            continue

        latest = periods[-1]
        avg_ratio = sum(p["ratio"] for p in periods) / len(periods)

        doc = (
            f"검색 트렌드: {title}. 키워드: {', '.join(keywords)}. "
            f"최근 검색량 비율: {latest['ratio']:.1f}, 평균: {avg_ratio:.1f}. "
            f"기간: {periods[0]['period']}~{latest['period']}"
        )
        doc_id = f"search_{hashlib.md5(doc.encode()).hexdigest()}"
        meta = {
            "type": "search_trend",
            "category": title,
            "keywords": ", ".join(keywords),
            "latest_ratio": latest["ratio"],
            "avg_ratio": avg_ratio,
            "date": data["collected_at"][:10],
            "source": "naver",
        }
        yield doc, doc_id, meta


def _build_naver_shopping_docs(fpath: str):
    """Yield (doc, doc_id, meta) tuples for naver shopping-trend file."""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [경고] 파일 읽기 실패 (건너뜀): {fpath} - {e}")
        return

    for result in data.get("results", []):
        title = result.get("title", "")
        periods = result.get("data", [])
        if not periods:
            continue

        latest = periods[-1]
        avg_ratio = sum(p["ratio"] for p in periods) / len(periods)

        doc = (
            f"쇼핑 트렌드: {data['description']}. 카테고리: {title}. "
            f"최근 쇼핑 비율: {latest['ratio']:.1f}, 평균: {avg_ratio:.1f}. "
            f"기간: {periods[0]['period']}~{latest['period']}"
        )
        doc_id = f"shopping_{hashlib.md5(doc.encode()).hexdigest()}"
        meta = {
            "type": "shopping_trend",
            "category": title,
            "latest_ratio": latest["ratio"],
            "avg_ratio": avg_ratio,
            "date": data["collected_at"][:10],
            "source": "naver",
        }
        yield doc, doc_id, meta


def _build_kosis_docs(fpath: str):
    """Yield (doc, doc_id, meta) tuples for a KOSIS / data_kr file."""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [경고] 파일 읽기 실패 (건너뜀): {fpath} - {e}")
        return

    name = data.get("name", "")
    desc = data.get("description", "")
    records = data.get("records", [])

    for rec in records:
        item_name = rec.get("ITM_NM", "")
        dim_name = rec.get("C1_NM", "")
        value = rec.get("DT", "")
        unit = rec.get("UNIT_NM", "")
        period = rec.get("PRD_DE", "")

        doc = (
            f"통계: {desc}. 항목: {item_name}, 분류: {dim_name}, "
            f"값: {value} {unit}, 기간: {period}"
        )
        doc_id = f"kosis_{hashlib.md5(doc.encode()).hexdigest()}"
        meta = {
            "type": "statistics",
            "category": name,
            "description": desc,
            "item": item_name,
            "dimension": dim_name,
            "value": str(value),
            "unit": unit,
            "period": period,
            "source": data.get("source", "kosis"),
        }
        yield doc, doc_id, meta


# ---------------------------------------------------------------------------
# Main indexing logic
# ---------------------------------------------------------------------------

def run_indexing(incremental: bool = False, dry_run: bool = False) -> None:
    metadata = load_metadata()
    last_indexed_ts: float = metadata.get("last_indexed_timestamp", 0.0) if incremental else 0.0
    mode_label = "증분 업데이트 (incremental)" if incremental else "전체 재구축 (full rebuild)"
    if dry_run:
        mode_label += " [DRY-RUN]"

    print("=" * 50)
    print(f"AIPA RAG 인덱스 구축 - {mode_label}")
    print("=" * 50)

    if incremental and last_indexed_ts:
        last_dt = datetime.fromtimestamp(last_indexed_ts, tz=timezone.utc)
        print(f"  마지막 인덱싱: {last_dt.isoformat()}")

    # Collect file paths per source -------------------------------------------
    naver_search_files = _get_json_files(str(DATA_DIR / "naver" / "search_trend_*_*.json"))
    naver_shopping_files = _get_json_files(str(DATA_DIR / "naver" / "shopping_*_*.json"))
    kosis_files = _get_json_files(str(DATA_DIR / "kosis" / "*.json")) + _get_json_files(str(DATA_DIR / "data_kr" / "*.json"))

    if incremental and last_indexed_ts:
        naver_search_files = _filter_modified_since(naver_search_files, last_indexed_ts)
        naver_shopping_files = _filter_modified_since(naver_shopping_files, last_indexed_ts)
        kosis_files = _filter_modified_since(kosis_files, last_indexed_ts)

    total_files = len(naver_search_files) + len(naver_shopping_files) + len(kosis_files)
    print(f"\n대상 파일 수: {total_files}건 (네이버 검색: {len(naver_search_files)}, "
          f"네이버 쇼핑: {len(naver_shopping_files)}, KOSIS: {len(kosis_files)})")

    if dry_run:
        if naver_search_files:
            print("\n[네이버 검색 트렌드 파일]")
            for fp in naver_search_files:
                print(f"  - {fp}")
        if naver_shopping_files:
            print("\n[네이버 쇼핑 트렌드 파일]")
            for fp in naver_shopping_files:
                print(f"  - {fp}")
        if kosis_files:
            print("\n[KOSIS/통계 파일]")
            for fp in kosis_files:
                print(f"  - {fp}")
        print("\n[DRY-RUN] 실제 인덱싱 없이 종료합니다.")
        return

    if total_files == 0:
        print("변경된 파일이 없습니다. 인덱싱을 건너뜁니다.")
        return

    # Embedding model ---------------------------------------------------------
    print("\n[1/3] 임베딩 모델 로드...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sroberta-multitask"
    )

    # ChromaDB ----------------------------------------------------------------
    client = chromadb.PersistentClient(path=DB_PATH)
    naver_col = client.get_or_create_collection("naver_trends", embedding_function=ef)
    kosis_col = client.get_or_create_collection("kosis_stats", embedding_function=ef)

    # Existing ID sets (for dedup) --------------------------------------------
    _existing_naver_ids = set(naver_col.get()["ids"]) if naver_col.count() > 0 else set()
    _existing_kosis_ids = set(kosis_col.get()["ids"]) if kosis_col.count() > 0 else set()

    # ========== 네이버 ==========
    print("\n[2/3] 네이버 데이터 인덱싱...")
    naver_docs, naver_ids, naver_metas = [], [], []

    for fpath in naver_search_files:
        for doc, doc_id, meta in _build_naver_search_docs(fpath):
            if doc_id not in _existing_naver_ids:
                naver_docs.append(doc)
                naver_ids.append(doc_id)
                naver_metas.append(meta)

    for fpath in naver_shopping_files:
        for doc, doc_id, meta in _build_naver_shopping_docs(fpath):
            if doc_id not in _existing_naver_ids:
                naver_docs.append(doc)
                naver_ids.append(doc_id)
                naver_metas.append(meta)

    if naver_docs:
        naver_col.add(documents=naver_docs, ids=naver_ids, metadatas=naver_metas)
        print(f"  네이버 트렌드: {len(naver_docs)}건 인덱싱 완료")
    else:
        print("  네이버 트렌드: 새로 추가할 문서 없음")

    # ========== KOSIS ==========
    print("\n[3/3] KOSIS/통계 데이터 인덱싱...")
    kosis_docs_list, kosis_ids_list, kosis_metas_list = [], [], []

    for fpath in kosis_files:
        for doc, doc_id, meta in _build_kosis_docs(fpath):
            if doc_id not in _existing_kosis_ids:
                kosis_docs_list.append(doc)
                kosis_ids_list.append(doc_id)
                kosis_metas_list.append(meta)

    if kosis_docs_list:
        kosis_col.add(documents=kosis_docs_list, ids=kosis_ids_list, metadatas=kosis_metas_list)
        print(f"  KOSIS/통계: {len(kosis_docs_list)}건 인덱싱 완료")
    else:
        print("  KOSIS/통계: 새로 추가할 문서 없음")

    total_indexed = len(naver_docs) + len(kosis_docs_list)
    now_iso = datetime.now(timezone.utc).isoformat()
    now_ts = datetime.now(timezone.utc).timestamp()

    # Update metadata ---------------------------------------------------------
    new_metadata = {
        "last_indexed_timestamp": now_ts,
        "last_indexed_iso": now_iso,
        "mode": "incremental" if incremental else "full",
        "stats": {
            "naver_new": len(naver_docs),
            "kosis_new": len(kosis_docs_list),
            "total_new": total_indexed,
            "naver_total": naver_col.count(),
            "kosis_total": kosis_col.count(),
        },
    }
    save_metadata(new_metadata)

    print(f"\n총 인덱싱: {total_indexed}건 (네이버 전체: {naver_col.count()}, KOSIS 전체: {kosis_col.count()})")
    print(f"DB 저장: {DB_PATH}")
    print(f"메타데이터 저장: {METADATA_PATH}")
    print("=" * 50)
    print("RAG 인덱스 구축 완료!")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIPA RAG 인덱스 구축")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="마지막 인덱싱 이후 변경된 파일만 처리 (증분 업데이트)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="인덱싱하지 않고 대상 파일만 출력",
    )
    args = parser.parse_args()
    run_indexing(incremental=args.incremental, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
