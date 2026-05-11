"""
JSONL 데이터 아카이브 스크립트

지정된 일수보다 오래된 JSONL 파일을 월별 gzip 아카이브로 압축.
최근 데이터는 그대로 유지.

사용법:
    python data/scripts/archive_data.py                    # 기본 30일
    python data/scripts/archive_data.py --days 60          # 60일 이전 파일 아카이브
    python data/scripts/archive_data.py --data-dir data/training --days 14
"""

import argparse
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def get_file_month(filepath: Path) -> str:
    """파일의 수정 시간에서 YYYY-MM 형식의 월 문자열 반환."""
    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
    return mtime.strftime("%Y-%m")


def find_old_jsonl_files(data_dir: Path, days: int) -> list[Path]:
    """지정 일수보다 오래된 .jsonl 파일 목록 반환. 이미 압축된 .gz 파일은 제외."""
    cutoff = datetime.now() - timedelta(days=days)
    old_files = []
    for f in data_dir.rglob("*.jsonl"):
        if f.stat().st_mtime < cutoff.timestamp():
            old_files.append(f)
    return sorted(old_files, key=lambda p: p.stat().st_mtime)


def archive_files(files: list[Path], archive_dir: Path, dry_run: bool = False) -> dict[str, list[Path]]:
    """
    파일들을 월별 gzip 아카이브로 압축.

    각 파일은 archive_dir/YYYY-MM/ 아래에 원본이름.jsonl.gz 로 저장.
    원본 파일은 압축 완료 후 삭제.

    Returns: {month: [archived_paths]}
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    for filepath in files:
        month = get_file_month(filepath)
        month_dir = archive_dir / month
        month_dir.mkdir(parents=True, exist_ok=True)

        gz_path = month_dir / (filepath.name + ".gz")

        if dry_run:
            print(f"  [DRY RUN] {filepath} -> {gz_path}")
        else:
            # gzip 압축
            with open(filepath, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            original_size = filepath.stat().st_size
            compressed_size = gz_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

            # 압축 성공 확인 후 원본 삭제
            if gz_path.exists() and gz_path.stat().st_size > 0:
                filepath.unlink()
                print(f"  {filepath.name} -> {gz_path.relative_to(archive_dir)} ({ratio:.0f}% smaller)")
            else:
                print(f"  WARNING: compression failed for {filepath.name}, keeping original")
                continue

        result.setdefault(month, []).append(gz_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="JSONL 히스토리 파일 월별 아카이브")
    parser.add_argument("--days", type=int, default=30, help="이 일수보다 오래된 파일을 아카이브 (기본: 30)")
    parser.add_argument("--data-dir", type=str, default="data/training", help="JSONL 파일이 있는 디렉토리")
    parser.add_argument("--archive-dir", type=str, default=None, help="아카이브 저장 디렉토리 (기본: data-dir/archive)")
    parser.add_argument("--dry-run", action="store_true", help="실제 작업 없이 대상 파일만 표시")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    archive_dir = Path(args.archive_dir) if args.archive_dir else data_dir / "archive"

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        return

    print(f"Data directory: {data_dir}")
    print(f"Archive directory: {archive_dir}")
    print(f"Archiving files older than {args.days} days")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)")
    print()

    old_files = find_old_jsonl_files(data_dir, args.days)

    if not old_files:
        print("No files to archive.")
        return

    print(f"Found {len(old_files)} file(s) to archive:")
    for f in old_files:
        age_days = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days
        print(f"  {f.name} ({age_days} days old, {f.stat().st_size / 1024:.1f} KB)")
    print()

    archived = archive_files(old_files, archive_dir, dry_run=args.dry_run)

    print(f"\nDone! Archived {sum(len(v) for v in archived.values())} files into {len(archived)} monthly folder(s).")


if __name__ == "__main__":
    main()
