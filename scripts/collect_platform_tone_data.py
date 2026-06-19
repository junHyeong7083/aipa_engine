"""
플랫폼별 톤 학습 데이터 수집 (Knowledge Distillation)

목적:
  Claude API에게 "이 사용자가 이 플랫폼에서 이 제품을 보면 어떻게 반응할까?"를
  플랫폼별 톤으로 답하게 시킨다. 그 답변을 모아 0.5B 모델 추가 학습 데이터로 사용.

기존 학습 데이터(training/data/training_data_deduped.jsonl)는
플랫폼 정보가 없는 일반 페르소나 평가 데이터.
이 스크립트는 그 위에 "플랫폼별 톤" 차원을 추가한다.

사용법:
  # 소량 테스트 (각 플랫폼당 3건씩 = 총 24건)
  python scripts/collect_platform_tone_data.py --per-platform 3

  # 본격 수집 (각 플랫폼당 50건씩 = 총 400건)
  python scripts/collect_platform_tone_data.py --per-platform 50 --concurrent 5
"""

import asyncio
import json
import os
import random
import argparse
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aipa_engine.platforms.platform_data import PLATFORM_PROFILES, SNSPlatform

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "platform_tone" / "platform_tone_data.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# 시뮬레이션할 자극물 (플랫폼별로 다른 카테고리 가능)
STIMULI = [
    {"category": "식품", "stimulus": "프리미엄 치킨 메뉴 '허니갈릭 치킨' 가격 18,000원"},
    {"category": "앱/서비스", "stimulus": "AI 가계부 앱 '머니로그' 월 4,900원, 영수증 자동 분류"},
    {"category": "광고", "stimulus": "갤럭시 Z 플립 신광고, 인플루언서가 카페에서 셀카 찍는 15초"},
    {"category": "콘텐츠", "stimulus": "넷플릭스 신작 '서울의 봄2' 12부작, 1980년대 정치 드라마"},
    {"category": "패션", "stimulus": "유니클로 에어리즘 오버사이즈 티 19,900원 신상"},
    {"category": "가전/전자", "stimulus": "무선 이어폰 ProAir 5세대 18만원, 노이즈캔슬링 신모델"},
    {"category": "정책", "stimulus": "정부의 청년 월세 지원 확대안 - 월 최대 30만원 지급"},
    {"category": "교육", "stimulus": "온라인 코딩 부트캠프 '코드캠프' 6개월 과정 990만원"},
    {"category": "부동산", "stimulus": "강남 신축 오피스텔 5억 5천만원, 역세권 위치"},
    {"category": "자동차", "stimulus": "현대 아이오닉 6 신차 5,400만원, 1회 충전 524km"},
]

# 페르소나 traits 풀 (플랫폼 traits와는 별개)
GENERAL_TRAITS = [
    "가성비 중시", "트렌드 민감", "건강 관심", "품질 중시",
    "실용적", "감성적", "보수적", "프리미엄 선호",
    "효율 중시", "자유로움", "워라밸", "자기계발",
    "SNS 활발", "또래 의식", "검소", "리스크 감수",
]


def build_persona(platform: SNSPlatform) -> dict:
    """플랫폼 사용자 특성을 반영한 페르소나 생성"""
    profile = PLATFORM_PROFILES[platform]

    # 연령
    age_items = list(profile.age_distribution.items())
    age = random.choices(
        [a for a, _ in age_items],
        weights=[w for _, w in age_items],
        k=1,
    )[0]

    # 성별
    gender = "male" if random.random() < profile.gender_ratio.get("male", 0.5) else "female"

    # 직업
    occupation = random.choice(profile.dominant_occupations)

    # 특성: 일반 2개 + 플랫폼 3개
    general = random.sample(GENERAL_TRAITS, 2)
    platform_traits = random.sample(profile.platform_traits, min(3, len(profile.platform_traits)))

    return {
        "age_group": age,
        "gender": gender,
        "occupation": occupation,
        "traits": general + platform_traits,
    }


def build_prompt(platform: SNSPlatform, persona: dict, stimulus: dict) -> str:
    """Claude에게 보낼 프롬프트 - 플랫폼 톤으로 응답 요청"""
    profile = PLATFORM_PROFILES[platform]
    traits_str = ", ".join(persona["traits"])
    gender_kr = "남성" if persona["gender"] == "male" else "여성"

    return f"""당신은 한국의 SNS 플랫폼 사용자 시뮬레이션 전문가입니다.

[플랫폼] {profile.name_kr}
[플랫폼 특성] {profile.description}
[톤 가이드] {profile.tone_guide}
[자주 다루는 주제] {', '.join(profile.common_topics)}

[페르소나]
- 연령대: {persona['age_group']}
- 성별: {gender_kr}
- 직업: {persona['occupation']}
- 특성: {traits_str}

[평가 대상]
- 카테고리: {stimulus['category']}
- 내용: {stimulus['stimulus']}

이 사용자가 {profile.name_kr}에서 이 콘텐츠를 봤을 때 작성할 법한 댓글/반응을
플랫폼 톤 그대로 1~2문장으로 작성하세요. 0~100점 평가 점수도 함께 매기세요.

JSON으로만 응답하세요:
{{
  "score": <0~100>,
  "reaction": "<플랫폼 톤의 1~2문장 반응>",
  "axis": "<호감도/구매의향/관심도 중 가장 적합한 것>"
}}"""


async def generate_one(
    client: anthropic.AsyncAnthropic,
    platform: SNSPlatform,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """1건 생성"""
    persona = build_persona(platform)
    stimulus = random.choice(STIMULI)
    prompt = build_prompt(platform, persona, stimulus)

    async with semaphore:
        try:
            resp = await client.messages.create(
                model=MODEL_NAME,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            text = resp.content[0].text.strip()
            # JSON 파싱
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                return None

            data = json.loads(text[start:end])
            score = data.get("score", 50)
            reaction = data.get("reaction", "")
            axis = data.get("axis", "관심도")

            if not reaction:
                return None

            return {
                "input": {
                    "task": "platform_reaction",
                    "platform": platform.value,
                    "platform_name": PLATFORM_PROFILES[platform].name_kr,
                    "stimulus": stimulus["stimulus"],
                    "stimulus_type": stimulus["category"],
                    "persona_age_group": persona["age_group"],
                    "persona_gender": persona["gender"],
                    "persona_occupation": persona["occupation"],
                    "persona_traits": persona["traits"],
                    "scores": [{"name": axis, "score": score}],
                },
                "output": {
                    "reasonings": [reaction],
                    "open_response": reaction,
                    "confidence": 0.8,
                },
            }
        except Exception as e:
            print(f"  [{platform.value}] 실패: {e}")
            return None


async def collect(per_platform: int, concurrent: int):
    """모든 플랫폼에 대해 데이터 수집"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY 가 .env 에 없음")
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrent)

    platforms = list(SNSPlatform)
    total = per_platform * len(platforms)

    print(f"수집 시작: 플랫폼 {len(platforms)}개 × {per_platform}건 = 총 {total}건")
    print(f"동시 호출: {concurrent}, 모델: {MODEL_NAME}")
    print(f"출력: {OUTPUT_PATH}")
    print()

    tasks = []
    for platform in platforms:
        for _ in range(per_platform):
            tasks.append(generate_one(client, platform, semaphore))

    # 진행률 표시
    results = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if result:
            results.append(result)
        if completed % 10 == 0 or completed == total:
            print(f"  진행: {completed}/{total} (성공 {len(results)})")

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n완료: {len(results)}/{total}건 수집")
    print(f"저장: {OUTPUT_PATH}")

    # 플랫폼별 카운트
    from collections import Counter
    counts = Counter(r["input"]["platform"] for r in results)
    print("\n[플랫폼별 수집량]")
    for p in platforms:
        n = counts.get(p.value, 0)
        print(f"  {PLATFORM_PROFILES[p].name_kr:<12} {n}건")

    # 샘플 출력
    if results:
        print("\n[샘플 3건]")
        for r in results[:3]:
            inp = r["input"]
            out = r["output"]
            print(f"  [{inp['platform_name']}] {inp['persona_age_group']} {inp['persona_occupation']}")
            print(f"    자극: {inp['stimulus'][:50]}")
            print(f"    점수: {inp['scores'][0]['score']} / 톤: {out['reasonings'][0][:100]}")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-platform", type=int, default=5, help="플랫폼당 수집할 데이터 수")
    parser.add_argument("--concurrent", type=int, default=3, help="동시 API 호출 수")
    args = parser.parse_args()

    asyncio.run(collect(args.per_platform, args.concurrent))


if __name__ == "__main__":
    main()
