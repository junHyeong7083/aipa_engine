"""
플랫폼 톤 데이터 + 기존 reasoning 데이터 통합

입력:
  - training/data/reasoning_only.jsonl  (기존 이유 생성 데이터)
  - data/platform_tone/platform_tone_data.jsonl  (Claude가 만든 플랫폼 톤 데이터)

출력:
  - training/data/reasoning_with_platform.jsonl  (통합 데이터셋)

기존 reasoning 데이터는 input에 task='evaluation' 으로 표시되고,
플랫폼 데이터는 task='platform_reaction'으로 표시되어 멀티태스크 학습 가능.
"""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

EXISTING_PATH = ROOT / "training" / "data" / "reasoning_only.jsonl"
PLATFORM_PATH = ROOT / "data" / "platform_tone" / "platform_tone_data.jsonl"
OUTPUT_PATH = ROOT / "training" / "data" / "reasoning_with_platform.jsonl"


def normalize_existing(item: dict) -> dict:
    """기존 reasoning_only 데이터에 task='evaluation' 추가"""
    if "input" in item and "task" not in item["input"]:
        item["input"]["task"] = "evaluation"
    return item


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_count = 0
    platform_count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        # 기존 데이터
        if EXISTING_PATH.exists():
            with open(EXISTING_PATH, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = normalize_existing(json.loads(line))
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        existing_count += 1
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"[경고] 기존 reasoning 데이터 없음: {EXISTING_PATH}")

        # 플랫폼 데이터
        if PLATFORM_PATH.exists():
            with open(PLATFORM_PATH, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        platform_count += 1
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"[경고] 플랫폼 데이터 없음: {PLATFORM_PATH}")

    total = existing_count + platform_count
    print(f"통합 완료")
    print(f"  기존 reasoning 데이터: {existing_count}건")
    print(f"  플랫폼 톤 데이터:      {platform_count}건")
    print(f"  합계:                  {total}건")
    print(f"  저장:                  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
