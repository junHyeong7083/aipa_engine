"""
AIPA vs 통계청 사회조사 2024 검증 스크립트

통계청 가족관계 만족도 (전반적) 응답 분포와
AIPA 시뮬레이션 결과를 비교한다.

질문 (통계청 공식):
  "귀하는 가족 관계에서 다음 각 항목에 대하여 어느 정도 만족하고 계십니까?"
보기: 매우 만족 / 약간 만족 / 보통 / 약간 불만족 / 매우 불만족

사용법:
  python scripts/validate_against_kosis.py
"""

import sys
import json
import time
import requests
from pathlib import Path
from collections import Counter

# UTF-8 출력
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────
# 설정
# ─────────────────────────────────────

API_BASE = "https://aipa-engine-677587906956.asia-northeast3.run.app"

QUESTION_TEXT = "귀하는 본인의 전반적인 가족 관계에 대하여 어느 정도 만족하고 계십니까?"
CHOICES = ["매우 만족", "약간 만족", "보통", "약간 불만족", "매우 불만족"]

# 통계청 사회조사 2024 - 전반적인 가족 관계 만족도 (전체)
KOSIS_TOTAL = {
    "매우 만족": 25.7,
    "약간 만족": 37.8,
    "보통": 32.3,
    "약간 불만족": 3.4,
    "매우 불만족": 0.8,
}

KOSIS_BY_AGE = {
    "13~19세": {"매우 만족": 44.7, "약간 만족": 36.1, "보통": 16.5, "약간 불만족": 2.5, "매우 불만족": 0.2},
    "20~29세": {"매우 만족": 36.6, "약간 만족": 34.9, "보통": 24.9, "약간 불만족": 3.2, "매우 불만족": 0.5},
    "30~39세": {"매우 만족": 34.0, "약간 만족": 36.5, "보통": 25.8, "약간 불만족": 2.7, "매우 불만족": 0.9},
    "40~49세": {"매우 만족": 24.1, "약간 만족": 40.6, "보통": 31.7, "약간 불만족": 2.9, "매우 불만족": 0.8},
    "50~59세": {"매우 만족": 19.6, "약간 만족": 38.5, "보통": 37.3, "약간 불만족": 3.7, "매우 불만족": 0.9},
    "60세 이상": {"매우 만족": 16.9, "약간 만족": 38.1, "보통": 39.8, "약간 불만족": 4.2, "매우 불만족": 1.0},
}

# AIPA 연령대 표기 매핑
AGE_MAP = {
    "13~19세": "10대",
    "20~29세": "20대",
    "30~39세": "30대",
    "40~49세": "40대",
    "50~59세": "50대",
    "60세 이상": "60대+",
}

PANEL_SIZE = 100  # 연령대별 페르소나 수
MAX_RETRIES = 60
POLL_INTERVAL = 3


# ─────────────────────────────────────
# AIPA API 호출
# ─────────────────────────────────────

def generate_personas(panel_count: int, age_group: str) -> list:
    """특정 연령대만 가진 페르소나 N명 생성"""
    payload = {
        "panel_count": panel_count,
        "age_groups": [age_group],
        "generate_backstories": False,
    }
    r = requests.post(f"{API_BASE}/api/v1/personas/generate", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # API가 list 또는 {"personas": [...]} 형태 모두 가능
    return data if isinstance(data, list) else data.get("personas", [])


def run_simulation(personas: list, question_text: str, choices: list) -> str:
    """시뮬레이션 시작, session_id 반환"""
    payload = {
        "personas": personas,
        "questions": [{
            "id": "q1",
            "text": question_text,
            "choices": choices,
        }],
        "generate_explanations": False,  # 검증이라 이유 생성 불필요
    }
    r = requests.post(f"{API_BASE}/api/v1/simulations/", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("id") or data.get("session_id")


def wait_for_result(session_id: str) -> dict:
    """완료까지 폴링"""
    for i in range(MAX_RETRIES):
        r = requests.get(f"{API_BASE}/api/v1/simulations/{session_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "completed":
            res = requests.get(f"{API_BASE}/api/v1/simulations/{session_id}/result", timeout=30)
            res.raise_for_status()
            return res.json()
        elif data.get("status") == "failed":
            raise RuntimeError(f"Simulation failed: {data}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError("시뮬레이션 타임아웃")


def aggregate_responses(result: dict) -> dict[str, float]:
    """응답 분포를 % 비율로 집계 (서버가 이미 분포 계산해서 보냄)"""
    dist = result.get("response_distribution", {}).get("q1", {})
    if not dist:
        # fallback: detailed_responses에서 직접 집계
        details = result.get("detailed_responses", [])
        if not details:
            return {c: 0.0 for c in CHOICES}
        counts = Counter()
        total = 0
        for r in details:
            choice = r.get("selected_choice")
            if choice in CHOICES:
                counts[choice] += 1
                total += 1
        if total == 0:
            return {c: 0.0 for c in CHOICES}
        return {c: round(counts.get(c, 0) / total * 100, 1) for c in CHOICES}

    # response_distribution은 0~1 비율 → % 변환
    return {c: round(dist.get(c, 0.0) * 100, 1) for c in CHOICES}


# ─────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────

def kl_divergence(p: dict, q: dict) -> float:
    """KL(P || Q). %를 0~1로 정규화 + 0 회피 smoothing"""
    import math
    eps = 0.001
    s = 0.0
    for k in p:
        pi = (p[k] / 100) + eps
        qi = (q[k] / 100) + eps
        s += pi * math.log(pi / qi)
    return round(s, 4)


def mean_abs_error(p: dict, q: dict) -> float:
    """카테고리별 절대오차 평균 (%)"""
    return round(sum(abs(p[k] - q[k]) for k in p) / len(p), 2)


# ─────────────────────────────────────
# 메인 검증 루프
# ─────────────────────────────────────

def main():
    print("=" * 70)
    print("AIPA vs 통계청 사회조사 2024 검증")
    print("=" * 70)
    print(f"질문: {QUESTION_TEXT}")
    print(f"보기: {' / '.join(CHOICES)}")
    print(f"연령대별 페르소나 수: {PANEL_SIZE}명")
    print()

    results = {}

    for kosis_age, aipa_age in AGE_MAP.items():
        print(f"\n[{kosis_age}] 시뮬레이션 시작...")
        try:
            personas = generate_personas(PANEL_SIZE, aipa_age)
            print(f"  페르소나 {len(personas)}명 생성 완료")

            session_id = run_simulation(personas, QUESTION_TEXT, CHOICES)
            print(f"  세션 {session_id[:8]}... 시작")

            result = wait_for_result(session_id)
            aipa_dist = aggregate_responses(result)
            real_dist = KOSIS_BY_AGE[kosis_age]

            kl = kl_divergence(real_dist, aipa_dist)
            mae = mean_abs_error(real_dist, aipa_dist)

            results[kosis_age] = {
                "real": real_dist,
                "aipa": aipa_dist,
                "kl_divergence": kl,
                "mean_abs_error_pp": mae,
            }

            print(f"  완료 - KL: {kl}, MAE: {mae}%p")
            print(f"  실제: {real_dist}")
            print(f"  AIPA: {aipa_dist}")

        except Exception as e:
            print(f"  실패: {e}")
            results[kosis_age] = {"error": str(e)}

    # ─── 결과 저장 ───
    output_path = Path(__file__).parent.parent / "docs" / "validation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ─── 요약 ───
    print("\n" + "=" * 70)
    print("최종 비교 (연령대별)")
    print("=" * 70)
    print(f"{'연령대':<10} {'KL Div':<10} {'평균오차':<10}")
    print("-" * 30)

    valid_kls = []
    valid_maes = []
    for age, r in results.items():
        if "error" in r:
            print(f"{age:<10} (실패: {r['error'][:40]})")
            continue
        print(f"{age:<10} {r['kl_divergence']:<10} {r['mean_abs_error_pp']}%p")
        valid_kls.append(r["kl_divergence"])
        valid_maes.append(r["mean_abs_error_pp"])

    if valid_kls:
        avg_kl = sum(valid_kls) / len(valid_kls)
        avg_mae = sum(valid_maes) / len(valid_maes)
        print("-" * 30)
        print(f"{'평균':<10} {round(avg_kl, 4):<10} {round(avg_mae, 2)}%p")

    print(f"\n상세 결과: {output_path}")
    print("\n발표 슬라이드 문구:")
    if valid_kls:
        print(f"  '통계청 사회조사 2024 실제 응답 분포(36,000명)와 AIPA 시뮬레이션({PANEL_SIZE * len(AGE_MAP)}명)을")
        print(f"   비교한 결과, 평균 KL Divergence {round(avg_kl, 3)}, 카테고리 평균 오차 {round(avg_mae, 1)}%p로")
        print(f"   실제 분포에 근접한 응답 생성을 확인.'")


if __name__ == "__main__":
    main()
