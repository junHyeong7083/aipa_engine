"""
AIPA vs 한국리서치 동영상 콘텐츠 조사 검증 (v2)

AIPA 학습 도메인(콘텐츠)에 매칭되는 질문으로 재검증.
실제 분포보다 "연령별 트렌드 방향성" 비교에 집중.

출처: 한국리서치 동영상 콘텐츠 이용행태 조사 (2023.10, 1,000명)
"""

import sys
import json
import time
import requests
from pathlib import Path
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

API_BASE = "https://aipa-engine-677587906956.asia-northeast3.run.app"

# 한국리서치 동영상 시청 빈도 (주 6일 이상 비율)
REAL_TREND = {
    "20대": 78,   # 18-29세
    "30대": 64,
    "40대": 61,
    "50대": 42,
    "60대+": 38,
}

# AIPA에 시뮬레이션할 질문 - 콘텐츠 카테고리 매칭
QUESTION_TEXT = "귀하는 OTT(넷플릭스, 디즈니플러스, 티빙 등) 또는 유튜브 같은 동영상 콘텐츠 서비스를 얼마나 자주 시청하십니까?"
CHOICES = [
    "거의 매일 시청",
    "주 4~6일 시청",
    "주 2~3일 시청",
    "주 1일 이하 시청",
    "거의 시청하지 않음",
]

# 첫 두 보기를 "헤비 시청자(주 4일+)"로 간주
HEAVY_VIEWER_CHOICES = ["거의 매일 시청", "주 4~6일 시청"]

PANEL_SIZE = 80


def generate_personas(panel_count, age_group):
    r = requests.post(
        f"{API_BASE}/api/v1/personas/generate",
        json={"panel_count": panel_count, "age_groups": [age_group], "generate_backstories": False},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get("personas", [])


def run_simulation(personas, q_text, choices):
    r = requests.post(
        f"{API_BASE}/api/v1/simulations/",
        json={
            "personas": personas,
            "questions": [{"id": "q1", "text": q_text, "choices": choices}],
            "generate_explanations": False,
        },
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()
    return d.get("id") or d.get("session_id")


def wait_for_result(session_id):
    for _ in range(60):
        r = requests.get(f"{API_BASE}/api/v1/simulations/{session_id}", timeout=30)
        if r.json().get("status") == "completed":
            res = requests.get(f"{API_BASE}/api/v1/simulations/{session_id}/result", timeout=30)
            return res.json()
        time.sleep(3)
    raise TimeoutError()


def heavy_viewer_pct(result):
    """주 4일 이상 시청 비율 계산"""
    dist = result.get("response_distribution", {}).get("q1", {})
    if dist:
        return round(sum(dist.get(c, 0) for c in HEAVY_VIEWER_CHOICES) * 100, 1)
    # fallback
    details = result.get("detailed_responses", [])
    if not details:
        return 0
    heavy = sum(1 for r in details if r.get("selected_choice") in HEAVY_VIEWER_CHOICES)
    return round(heavy / len(details) * 100, 1)


def main():
    print("=" * 70)
    print("AIPA vs 한국리서치 동영상 콘텐츠 조사 검증")
    print("=" * 70)
    print(f"질문: {QUESTION_TEXT}")
    print(f"검증 지표: 주 4일 이상 시청 비율 (헤비 시청자)")
    print()

    results = {}
    for age in REAL_TREND:
        print(f"\n[{age}] 시뮬레이션...")
        try:
            personas = generate_personas(PANEL_SIZE, age)
            sid = run_simulation(personas, QUESTION_TEXT, CHOICES)
            res = wait_for_result(sid)
            aipa_pct = heavy_viewer_pct(res)
            real_pct = REAL_TREND[age]
            diff = round(aipa_pct - real_pct, 1)
            results[age] = {"real": real_pct, "aipa": aipa_pct, "diff_pp": diff}
            print(f"  실제: {real_pct}% / AIPA: {aipa_pct}% / 차이: {diff:+}%p")
        except Exception as e:
            print(f"  실패: {e}")
            results[age] = {"error": str(e)}

    # 저장
    output = Path(__file__).parent.parent / "docs" / "validation_v2_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 요약
    print("\n" + "=" * 70)
    print("최종 비교 (헤비 시청자 비율)")
    print("=" * 70)
    print(f"{'연령':<8} {'실제':<10} {'AIPA':<10} {'차이':<10}")
    print("-" * 40)

    valid = [(age, r) for age, r in results.items() if "error" not in r]
    if valid:
        for age, r in valid:
            print(f"{age:<8} {r['real']}%{'':<6} {r['aipa']}%{'':<6} {r['diff_pp']:+}%p")

        # 트렌드 방향성 검증
        real_seq = [r["real"] for _, r in valid]
        aipa_seq = [r["aipa"] for _, r in valid]

        # 단조 감소 여부
        real_monotonic = all(real_seq[i] >= real_seq[i+1] for i in range(len(real_seq)-1))
        aipa_monotonic = all(aipa_seq[i] >= aipa_seq[i+1] for i in range(len(aipa_seq)-1))

        # 상관계수 (Pearson)
        n = len(real_seq)
        mean_r = sum(real_seq) / n
        mean_a = sum(aipa_seq) / n
        cov = sum((real_seq[i] - mean_r) * (aipa_seq[i] - mean_a) for i in range(n))
        var_r = sum((x - mean_r) ** 2 for x in real_seq) ** 0.5
        var_a = sum((x - mean_a) ** 2 for x in aipa_seq) ** 0.5
        corr = round(cov / (var_r * var_a + 1e-9), 3) if var_r * var_a > 0 else 0

        mae = round(sum(abs(r["diff_pp"]) for _, r in valid) / len(valid), 1)

        print("-" * 40)
        print(f"\n[트렌드 방향성]")
        print(f"  실제 단조 감소: {'O' if real_monotonic else 'X'}")
        print(f"  AIPA 단조 감소: {'O' if aipa_monotonic else 'X'}")
        print(f"  Pearson 상관계수: {corr}")
        print(f"  평균 절대 오차: {mae}%p")

        print("\n[발표 슬라이드 문구]")
        print(f"  한국리서치 동영상 콘텐츠 이용행태 조사(2023, 1,000명)와 비교한 결과,")
        print(f"  AIPA는 연령대별 시청 빈도 트렌드를 Pearson 상관계수 {corr}으로 재현했으며,")
        print(f"  평균 오차는 {mae}%p였습니다.")

        return corr, mae

    return None, None


if __name__ == "__main__":
    main()
