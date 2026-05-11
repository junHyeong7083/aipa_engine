"""
설문 응답 데이터를 임베딩 모델 학습 포맷으로 변환

survey_responses.jsonl → 임베딩 모델이 이해하는 형태

변환 내용:
1. 보기 선택 → 0~100 점수로 변환
2. 질문 카테고리 매핑
3. 축 자동 결정
4. 기존 평가 데이터와 합쳐서 통합 학습 데이터 생성

사용법:
  python training/convert_survey_to_embedding.py
"""

import json
from pathlib import Path

SURVEY_PATH = Path(__file__).parent / "data" / "survey_responses.jsonl"
EVAL_PATH = Path(__file__).parent / "data" / "training_data_deduped.jsonl"
OUTPUT_PATH = Path(__file__).parent / "data" / "combined_embedding_data.jsonl"

# 설문 카테고리 → 임베딩 카테고리 매핑
CATEGORY_MAP = {
    "소비/쇼핑": "앱/서비스",
    "미디어/콘텐츠": "콘텐츠",
    "건강/웰빙": "식품",
    "직장/커리어": "기타",
    "기술/디지털": "앱/서비스",
    "사회/정치": "정책",
    "교육": "교육",
    "주거/부동산": "부동산",
    "환경": "기타",
    "가족/관계": "기타",
}

# 질문 카테고리 → 축 매핑
CATEGORY_AXIS_MAP = {
    "소비/쇼핑": "구매의향",
    "미디어/콘텐츠": "관심도",
    "건강/웰빙": "관심도",
    "직장/커리어": "호감도",
    "기술/디지털": "사용의향",
    "사회/정치": "관심도",
    "교육": "관심도",
    "주거/부동산": "관심도",
    "환경": "관심도",
    "가족/관계": "호감도",
}


def choice_to_score(selected: str, choices: list[str]) -> int:
    """
    선택한 보기를 0~100 점수로 변환

    첫번째 보기 = 긍정 (높은 점수)
    마지막 보기 = 부정 (낮은 점수)
    """
    if selected not in choices:
        return 50

    idx = choices.index(selected)
    n = len(choices)

    if n == 1:
        return 50

    # 보기 순서에 따라 점수 배분
    # 첫번째 = 90, 마지막 = 10, 나머지 균등 분배
    score = 90 - (idx * 80 / (n - 1))
    return max(0, min(100, int(score)))


def convert_survey_item(item: dict) -> dict | None:
    """설문 1건을 임베딩 학습 포맷으로 변환"""
    inp = item.get("input", {})
    out = item.get("output", {})

    category = inp.get("category", "")
    selected = out.get("selected", "")
    choices = inp.get("choices", [])

    if not selected or not choices:
        return None

    # 점수 변환
    score = choice_to_score(selected, choices)

    # 카테고리 매핑
    mapped_category = CATEGORY_MAP.get(category, "기타")
    axis = CATEGORY_AXIS_MAP.get(category, "관심도")

    return {
        "input": {
            "stimulus": inp.get("question", ""),
            "stimulus_type": mapped_category,
            "persona_age_group": inp.get("persona_age_group", ""),
            "persona_gender": inp.get("persona_gender", ""),
            "persona_occupation": inp.get("persona_occupation", ""),
            "persona_income": inp.get("persona_income", ""),
            "persona_traits": inp.get("persona_traits", []),
            "axes": [axis],
        },
        "output": {
            "evaluations": [
                {
                    "name": axis,
                    "score": score,
                    "reasoning": out.get("reasoning", ""),
                }
            ],
            "open_response": out.get("reasoning", ""),
            "confidence": out.get("confidence", 0.7),
        },
    }


def main():
    print("설문 데이터 → 임베딩 학습 포맷 변환")
    print("=" * 50)

    # 1. 기존 평가 데이터 복사
    eval_count = 0
    combined = []

    print(f"\n[1/3] 기존 평가 데이터 로드: {EVAL_PATH}")
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                combined.append(line.strip())
                eval_count += 1
    print(f"  {eval_count}건 로드")

    # 2. 설문 데이터 변환
    survey_count = 0
    skipped = 0

    print(f"\n[2/3] 설문 데이터 변환: {SURVEY_PATH}")
    with open(SURVEY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                converted = convert_survey_item(item)
                if converted:
                    combined.append(json.dumps(converted, ensure_ascii=False))
                    survey_count += 1
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    print(f"  {survey_count}건 변환, {skipped}건 스킵")

    # 3. 통합 파일 저장
    print(f"\n[3/3] 통합 데이터 저장: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in combined:
            f.write(line + "\n")

    total = eval_count + survey_count
    print(f"\n완료!")
    print(f"  평가 데이터: {eval_count}건")
    print(f"  설문 데이터: {survey_count}건")
    print(f"  합계: {total}건")
    print(f"  저장: {OUTPUT_PATH}")
    print(f"\n다음 단계:")
    print(f"  python training/train_embedding.py --data training/data/combined_embedding_data.jsonl")


if __name__ == "__main__":
    main()
