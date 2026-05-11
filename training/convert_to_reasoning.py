"""
학습 데이터 변환: 점수를 입력으로 이동, 출력은 이유만

변환 전 (기존):
  입력: 페르소나 + 자극물
  출력: {evaluations: [{name, score, reasoning}], open_response, confidence}

변환 후 (새):
  입력: 페르소나 + 자극물 + 축별 점수
  출력: {reasonings: ["이유1", "이유2", ...], open_response, confidence}

사용법:
  python training/convert_to_reasoning.py
"""

import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent / "data" / "training_data_deduped.jsonl"
OUTPUT_PATH = Path(__file__).parent / "data" / "reasoning_only.jsonl"


def convert_example(item: dict) -> dict | None:
    """한 건의 학습 데이터를 변환"""
    inp = item.get("input", {})
    out = item.get("output", {})
    evaluations = out.get("evaluations", [])

    if not evaluations:
        return None

    # 축별 점수를 입력에 추가
    score_info = []
    reasonings = []
    for ev in evaluations:
        name = ev.get("name", "")
        score = ev.get("score", 50)
        reasoning = ev.get("reasoning", "")

        if not name or not reasoning:
            continue

        score_info.append({"name": name, "score": score})
        reasonings.append(reasoning)

    if not reasonings:
        return None

    return {
        "input": {
            "stimulus": inp.get("stimulus", ""),
            "stimulus_type": inp.get("stimulus_type", ""),
            "persona_age_group": inp.get("persona_age_group", ""),
            "persona_gender": inp.get("persona_gender", ""),
            "persona_occupation": inp.get("persona_occupation", ""),
            "persona_income": inp.get("persona_income", ""),
            "persona_traits": inp.get("persona_traits", []),
            "scores": score_info,  # 점수가 입력으로 이동
        },
        "output": {
            "reasonings": reasonings,  # 이유만 출력
            "open_response": out.get("open_response", ""),
            "confidence": out.get("confidence", 0.7),
        },
    }


def main():
    print("학습 데이터 변환: 점수 → 입력, 이유만 → 출력")
    print(f"입력: {INPUT_PATH}")
    print(f"출력: {OUTPUT_PATH}")

    converted = 0
    skipped = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                result = convert_example(item)
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    converted += 1
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    print(f"\n완료! {converted}건 변환, {skipped}건 스킵")
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
