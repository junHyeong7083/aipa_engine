"""
실제 크롤링/API 데이터를 0.5B 모델 학습 데이터로 변환

입력:
  - data/platform_tone/youtube_real_comments.jsonl   (유튜브 실제 댓글)
  - data/platform_tone/dcinside_real_posts.jsonl     (디시 실제 게시글)
  - data/platform_tone/platform_tone_data.jsonl      (Claude 합성 - 보강용)
  - training/data/reasoning_only.jsonl               (기존 평가 데이터)

출력:
  - training/data/reasoning_real_combined.jsonl

실제 데이터는 "사용자 자유 텍스트" 형식이므로,
플랫폼 + 카테고리 + 텍스트만 가지고 도메인 적응(domain adaptation) 학습용 포맷으로 변환:
  "이 페르소나가 [플랫폼]에서 [카테고리] 콘텐츠를 보면 어떻게 반응할지 작성하시오"
  → 실제 사용자가 쓴 텍스트가 정답
"""

import json
import sys
import random
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

EXISTING_REASONING = ROOT / "training" / "data" / "reasoning_only.jsonl"
YOUTUBE_RAW = ROOT / "data" / "platform_tone" / "youtube_real_comments.jsonl"
DCINSIDE_RAW = ROOT / "data" / "platform_tone" / "dcinside_real_posts.jsonl"
NAVER_RAW = ROOT / "data" / "platform_tone" / "naver_real_posts.jsonl"
CLAUDE_SYNTHETIC = ROOT / "data" / "platform_tone" / "platform_tone_data.jsonl"

OUTPUT = ROOT / "training" / "data" / "reasoning_real_combined.jsonl"


# 임시 페르소나 풀 (실제 데이터에 페르소나 정보 없으므로 플랫폼 프로필에서 랜덤 샘플)
DEFAULT_TRAITS = {
    "youtube": ["알고리즘 의존", "댓글 토론", "콘텐츠 충성도", "리뷰 신뢰"],
    "dcinside": ["익명적", "냉소적", "솔직함", "주류 반감", "유머/풍자"],
    "naver": ["정보 추구", "후기 신뢰", "꼼꼼한 비교", "카페 활동", "실용적"],
}

DEFAULT_AGE_BY_PLATFORM = {
    "youtube": ["20대", "30대", "40대"],
    "dcinside": ["20대", "30대"],
    "naver": ["30대", "40대", "50대"],
}

DEFAULT_GENDER_BY_PLATFORM = {
    "youtube": ["male", "female"],
    "dcinside": ["male", "male", "male", "female"],  # 78:22 비율
    "naver": ["female", "female", "male"],  # 네이버 카페/블로그는 여성 비중 높음
}


def convert_youtube_comment(item: dict) -> dict:
    """유튜브 댓글 → 학습 데이터 포맷"""
    age = random.choice(DEFAULT_AGE_BY_PLATFORM["youtube"])
    gender = random.choice(DEFAULT_GENDER_BY_PLATFORM["youtube"])

    return {
        "input": {
            "task": "platform_reaction",
            "platform": "youtube",
            "platform_name": "유튜브",
            "stimulus": f"{item.get('category', '콘텐츠')} 관련 동영상 ({item.get('search_keyword', '')})",
            "stimulus_type": item.get("category", "콘텐츠"),
            "persona_age_group": age,
            "persona_gender": gender,
            "persona_occupation": "직장인(대리)",
            "persona_traits": DEFAULT_TRAITS["youtube"][:3],
            "scores": [{"name": "관심도", "score": 70}],
            "source_type": "real_youtube_comment",
        },
        "output": {
            "reasonings": [item["text"]],
            "open_response": item["text"],
            "confidence": 0.95,
        },
    }


def convert_naver_post(item: dict) -> dict:
    """네이버 카페/블로그/지식인 게시글 → 학습 데이터 포맷"""
    age = random.choice(DEFAULT_AGE_BY_PLATFORM["naver"])
    gender = random.choice(DEFAULT_GENDER_BY_PLATFORM["naver"])

    title = item.get("title", "").strip()
    content = item.get("content", "").strip()
    text = (title + ". " + content) if title and content else (title or content)

    if not text or len(text) < 15:
        return None

    source = item.get("source", "naver")  # naver_cafe / naver_blog / naver_kin

    return {
        "input": {
            "task": "platform_reaction",
            "platform": "naver",
            "platform_name": "네이버",
            "stimulus": f"{item.get('category', '기타')} 관련 검색 ({item.get('search_keyword', '')})",
            "stimulus_type": item.get("category", "기타"),
            "persona_age_group": age,
            "persona_gender": gender,
            "persona_occupation": "주부" if gender == "female" else "직장인(대리)",
            "persona_traits": DEFAULT_TRAITS["naver"][:3],
            "scores": [{"name": "관심도", "score": 65}],
            "source_type": f"real_{source}",
        },
        "output": {
            "reasonings": [text[:500]],
            "open_response": text[:500],
            "confidence": 0.95,
        },
    }


def convert_dcinside_post(item: dict) -> dict:
    """디시 게시글 → 학습 데이터 포맷"""
    age = random.choice(DEFAULT_AGE_BY_PLATFORM["dcinside"])
    gender = random.choice(DEFAULT_GENDER_BY_PLATFORM["dcinside"])

    # 본문 있으면 사용, 없으면 제목만
    text = item.get("content", "").strip() or item.get("title", "").strip()
    if not text or len(text) < 10:
        return None

    return {
        "input": {
            "task": "platform_reaction",
            "platform": "dcinside",
            "platform_name": "디시인사이드",
            "stimulus": f"{item.get('category', '기타')} 관련 게시글 (갤러리: {item.get('gallery', 'dcbest')})",
            "stimulus_type": item.get("category", "기타"),
            "persona_age_group": age,
            "persona_gender": gender,
            "persona_occupation": "직장인(신입)" if age == "20대" else "직장인(대리)",
            "persona_traits": DEFAULT_TRAITS["dcinside"][:3],
            "scores": [{"name": "관심도", "score": 60}],
            "source_type": "real_dcinside_post",
        },
        "output": {
            "reasonings": [text[:500]],
            "open_response": text[:500],
            "confidence": 0.95,
        },
    }


def main():
    random.seed(42)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    all_items = []
    counts = {"existing_evaluation": 0, "claude_synthetic_platform": 0,
              "real_youtube": 0, "real_dcinside": 0, "real_naver": 0}

    # 1. 기존 reasoning 데이터 (task=evaluation)
    if EXISTING_REASONING.exists():
        with open(EXISTING_REASONING, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if "task" not in item.get("input", {}):
                        item["input"]["task"] = "evaluation"
                    all_items.append(item)
                    counts["existing_evaluation"] += 1
                except json.JSONDecodeError:
                    continue

    # 2. Claude 합성 플랫폼 데이터
    if CLAUDE_SYNTHETIC.exists():
        with open(CLAUDE_SYNTHETIC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    item["input"]["source_type"] = "claude_synthetic"
                    all_items.append(item)
                    counts["claude_synthetic_platform"] += 1
                except json.JSONDecodeError:
                    continue

    # 3. 진짜 유튜브 댓글
    if YOUTUBE_RAW.exists():
        with open(YOUTUBE_RAW, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    converted = convert_youtube_comment(raw)
                    if converted:
                        all_items.append(converted)
                        counts["real_youtube"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    # 4. 진짜 디시 게시글
    if DCINSIDE_RAW.exists():
        with open(DCINSIDE_RAW, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    converted = convert_dcinside_post(raw)
                    if converted:
                        all_items.append(converted)
                        counts["real_dcinside"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    # 5. 진짜 네이버 카페/블로그/지식인
    if NAVER_RAW.exists():
        with open(NAVER_RAW, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    converted = convert_naver_post(raw)
                    if converted:
                        all_items.append(converted)
                        counts["real_naver"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    # 셔플 + 저장
    random.shuffle(all_items)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("통합 학습 데이터셋 빌드 완료")
    print("=" * 60)
    print(f"  기존 평가 데이터 (evaluation):       {counts['existing_evaluation']}건")
    print(f"  Claude 합성 플랫폼 데이터:           {counts['claude_synthetic_platform']}건")
    print(f"  유튜브 실제 댓글:                    {counts['real_youtube']}건  ⭐ 실데이터")
    print(f"  디시인사이드 실제 게시글:            {counts['real_dcinside']}건  ⭐ 실데이터")
    print(f"  네이버 카페/블로그/지식인:           {counts['real_naver']}건  ⭐ 실데이터")
    print(f"  ──────────────────────────────────")
    print(f"  합계:                                 {len(all_items)}건")
    print(f"  저장:                                 {OUTPUT}")
    print()
    print("다음:")
    print("  python training/train_reasoning_v2.py --data training/data/reasoning_real_combined.jsonl")


if __name__ == "__main__":
    main()
