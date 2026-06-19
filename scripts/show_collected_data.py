"""
수집한 진짜 데이터 발표용 미리보기

유튜브 댓글, 디시 게시글의 통계 + 샘플 출력.
발표 슬라이드에 그대로 캡처해서 쓸 수 있도록 정렬됨.

사용법:
  python scripts/show_collected_data.py
  python scripts/show_collected_data.py --samples 10
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent.parent
YT = ROOT / "data" / "platform_tone" / "youtube_real_comments.jsonl"
DC = ROOT / "data" / "platform_tone" / "dcinside_real_posts.jsonl"
CLAUDE = ROOT / "data" / "platform_tone" / "platform_tone_data.jsonl"


def load(path: Path):
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def show_youtube(items: list, samples: int):
    print("=" * 70)
    print(f"유튜브 실제 댓글 — 총 {len(items)}건")
    print("=" * 70)

    cats = Counter(i.get("category", "?") for i in items)
    print("\n[카테고리별]")
    for cat, n in cats.most_common():
        print(f"  {cat:<14} {n}건")

    print(f"\n[샘플 {samples}건]")
    for i, item in enumerate(items[:samples]):
        text = item.get("text", "").replace("\n", " ")[:120]
        cat = item.get("category", "?")
        votes = item.get("votes", "0")
        print(f"  {i+1:2d}. [{cat}] 👍{votes}")
        print(f"      {text}")
    print()


def show_dcinside(items: list, samples: int):
    print("=" * 70)
    print(f"디시인사이드 실제 게시글 — 총 {len(items)}건")
    print("=" * 70)

    galls = Counter(i.get("gallery", "?") for i in items)
    print("\n[갤러리별]")
    for g, n in galls.most_common():
        print(f"  {g:<14} {n}건")

    print(f"\n[샘플 {samples}건]")
    for i, item in enumerate(items[:samples]):
        title = item.get("title", "")[:80]
        content = (item.get("content") or "").replace("\n", " ")[:120]
        rec = item.get("recommend", item.get("recommend_list", "0"))
        gall = item.get("gallery", "?")
        print(f"  {i+1:2d}. [{gall}] 👍{rec}  {title}")
        if content:
            print(f"      └ {content}")
    print()


def show_claude_synthetic(items: list, samples: int):
    print("=" * 70)
    print(f"Claude 합성 플랫폼 톤 — 총 {len(items)}건")
    print("=" * 70)

    plats = Counter(i.get("input", {}).get("platform", "?") for i in items)
    print("\n[플랫폼별]")
    for p, n in plats.most_common():
        print(f"  {p:<12} {n}건")

    print(f"\n[샘플 {samples}건]")
    for i, item in enumerate(items[:samples]):
        inp = item.get("input", {})
        out = item.get("output", {})
        reaction = (out.get("reasonings", [""])[0] if out.get("reasonings") else "")[:120]
        plat = inp.get("platform_name", inp.get("platform", "?"))
        age = inp.get("persona_age_group", "?")
        score = (inp.get("scores", [{}])[0] or {}).get("score", "?")
        print(f"  {i+1:2d}. [{plat}] {age} / 점수 {score}")
        print(f"      {reaction}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="각 출처별로 보여줄 샘플 수")
    args = parser.parse_args()

    yt = load(YT)
    dc = load(DC)
    cl = load(CLAUDE)

    print()
    print("█" * 70)
    print("  AIPA - 수집 데이터 현황")
    print("█" * 70)
    print()
    print(f"  유튜브 실제 댓글:        {len(yt):>5}건")
    print(f"  디시인사이드 게시글:     {len(dc):>5}건")
    print(f"  Claude 합성 플랫폼 톤:   {len(cl):>5}건")
    print(f"  ─────────────────────────────")
    print(f"  플랫폼 데이터 합계:      {len(yt)+len(dc)+len(cl):>5}건")
    print()

    if yt:
        show_youtube(yt, args.samples)
    if dc:
        show_dcinside(dc, args.samples)
    if cl:
        show_claude_synthetic(cl, args.samples)


if __name__ == "__main__":
    main()
