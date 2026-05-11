"""
0.5B 모델 재학습용 통합 데이터 생성

기존 이유 생성 데이터 + 설문 응답 데이터 → 멀티태스크 학습 데이터

태스크 1: 평가 이유 생성 (기존)
  입력: 페르소나 + 자극물 + 점수
  출력: 이유 텍스트

태스크 2: 설문 답변 (새로 추가)
  입력: 페르소나 + 질문 + 보기 + 점수(임베딩 모델)
  출력: 보기 선택 + 이유

사용법:
  python training/combine_reasoning_data.py
"""

import json
from pathlib import Path

REASONING_PATH = Path(__file__).parent / "data" / "reasoning_only.jsonl"
SURVEY_PATH = Path(__file__).parent / "data" / "survey_responses.jsonl"
OUTPUT_PATH = Path(__file__).parent / "data" / "combined_reasoning.jsonl"


def convert_survey_for_reasoning(item: dict) -> dict | None:
    """설문 데이터를 0.5B 학습 포맷으로 변환"""
    inp = item.get("input", {})
    out = item.get("output", {})

    selected = out.get("selected", "")
    reasoning = out.get("reasoning", "")
    choices = inp.get("choices", [])

    if not selected or not reasoning or not choices:
        return None

    # 선택한 보기 → 점수 변환 (임베딩 모델과 동일한 방식)
    if selected in choices:
        idx = choices.index(selected)
        n = len(choices)
        score = int(90 - (idx * 80 / max(n - 1, 1)))
    else:
        score = 50

    # 보기를 점수 형태로 변환
    scores = [{"name": "응답", "score": score}]

    return {
        "input": {
            "stimulus": inp.get("question", ""),
            "stimulus_type": inp.get("category", "설문"),
            "persona_age_group": inp.get("persona_age_group", ""),
            "persona_gender": inp.get("persona_gender", ""),
            "persona_occupation": inp.get("persona_occupation", ""),
            "persona_income": inp.get("persona_income", ""),
            "persona_traits": inp.get("persona_traits", []),
            "scores": scores,
            # 설문 전용 필드
            "task": "survey",
            "choices": choices,
        },
        "output": {
            "reasonings": [reasoning],
            "selected": selected,
            "open_response": reasoning,
            "confidence": out.get("confidence", 0.7),
        },
    }


def main():
    print("0.5B 모델 통합 학습 데이터 생성")
    print("=" * 50)

    combined = []

    # 1. 기존 이유 생성 데이터
    reasoning_count = 0
    print(f"\n[1/3] 이유 생성 데이터: {REASONING_PATH}")
    with open(REASONING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # task 필드 추가
                item["input"]["task"] = "evaluation"
                combined.append(json.dumps(item, ensure_ascii=False))
                reasoning_count += 1
    print(f"  {reasoning_count}건")

    # 2. 설문 응답 데이터 변환
    survey_count = 0
    skipped = 0
    print(f"\n[2/3] 설문 응답 데이터: {SURVEY_PATH}")
    with open(SURVEY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                converted = convert_survey_for_reasoning(item)
                if converted:
                    combined.append(json.dumps(converted, ensure_ascii=False))
                    survey_count += 1
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    print(f"  {survey_count}건 변환, {skipped}건 스킵")

    # 3. 저장
    print(f"\n[3/3] 통합 데이터 저장: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in combined:
            f.write(line + "\n")

    total = reasoning_count + survey_count
    print(f"\n완료!")
    print(f"  평가 이유: {reasoning_count}건")
    print(f"  설문 응답: {survey_count}건")
    print(f"  합계: {total}건")
    print(f"\n다음 단계:")
    print(f"  python training/train_reasoning.py --data training/data/combined_reasoning.jsonl")


if __name__ == "__main__":
    main()
