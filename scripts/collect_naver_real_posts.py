"""
네이버 검색 API로 실제 카페/블로그/지식인 게시글 수집

공식 API: https://developers.naver.com/docs/serviceapi/search/
- /v1/search/cafearticle.json : 카페 게시글
- /v1/search/blog.json : 블로그 글
- /v1/search/kin.json : 지식iN Q&A

수집 데이터:
  - 실제 네이버 사용자가 작성한 글
  - 합법 (공식 API)
  - 무료 (일 25,000건 호출까지)

출력:
  data/platform_tone/naver_real_posts.jsonl

사용법:
  python scripts/collect_naver_real_posts.py
  python scripts/collect_naver_real_posts.py --keywords "신상 치킨" "유기농 샐러드"
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from html import unescape
import re

from dotenv import load_dotenv
load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "platform_tone" / "naver_real_posts.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# AIPA 학습 카테고리와 매칭되는 검색 키워드
KEYWORDS_BY_CATEGORY = {
    "식품": ["프리미엄 치킨 리뷰", "배달 도시락 후기", "샐러드 구독 서비스 후기", "신상 음료 리뷰"],
    "화장품": ["선크림 추천 후기", "립틴트 후기", "남자 스킨케어 추천", "쿠션팩트 리뷰"],
    "앱/서비스": ["가계부 앱 추천", "AI 영어회화 앱 후기", "독서 요약 서비스 후기"],
    "광고": ["삼성 갤럭시 광고 평가", "쿠팡 광고 후기", "당근마켓 광고 의견"],
    "콘텐츠": ["넷플릭스 신작 후기", "유튜브 채널 추천", "OTT 추천"],
    "패션": ["유니클로 신상 후기", "오버사이즈 티 추천", "올해 트렌드 패션"],
    "가전/전자": ["무선 이어폰 추천", "노이즈캔슬링 후기", "삼성 갤럭시 후기"],
    "교육": ["코딩 부트캠프 후기", "온라인 강의 추천", "독학 영어 후기"],
}

# 네이버 검색 API 엔드포인트
ENDPOINTS = {
    "cafe": "https://openapi.naver.com/v1/search/cafearticle.json",
    "blog": "https://openapi.naver.com/v1/search/blog.json",
    "kin": "https://openapi.naver.com/v1/search/kin.json",
}


def clean_html(text: str) -> str:
    """HTML 태그/엔티티 제거"""
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    return text.strip()


def search(endpoint_name: str, query: str, display: int = 10) -> list[dict]:
    """네이버 검색 API 호출"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("네이버 API 키 없음")
        return []

    url = ENDPOINTS[endpoint_name]
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": min(display, 100),
        "sort": "sim",  # 정확도 순
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            print(f"  [{endpoint_name}] HTTP {r.status_code}: {r.text[:100]}")
            return []
        return r.json().get("items", [])
    except Exception as e:
        print(f"  [{endpoint_name}] 실패: {e}")
        return []


def collect_for_category(category: str, keywords: list[str], per_keyword: int = 5) -> list[dict]:
    """카테고리 하나에 대해 카페/블로그/지식인 모두 수집"""
    collected = []

    for kw in keywords:
        print(f"  검색: '{kw}'")

        for source_name in ["cafe", "blog", "kin"]:
            items = search(source_name, kw, display=per_keyword)
            print(f"    [{source_name}] {len(items)}건")

            for item in items:
                title = clean_html(item.get("title", ""))
                description = clean_html(item.get("description", ""))

                # 너무 짧거나 광고성은 제외
                if len(description) < 20:
                    continue

                collected.append({
                    "source": f"naver_{source_name}",
                    "platform": "naver",
                    "category": category,
                    "search_keyword": kw,
                    "title": title,
                    "content": description,
                    "url": item.get("link", ""),
                    "metadata": {
                        "post_date": item.get("postdate") or item.get("bloggerlink"),
                        "blogger_name": item.get("bloggername"),
                        "cafe_name": item.get("cafename"),
                    },
                    "collected_at": datetime.utcnow().isoformat(),
                })

    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-keyword", type=int, default=5, help="키워드당 수집할 결과 수 (소스별)")
    parser.add_argument("--keywords", nargs="*", default=None, help="직접 키워드 지정 시")
    parser.add_argument("--category", type=str, default="콘텐츠", help="--keywords 사용 시 카테고리")
    args = parser.parse_args()

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("ERROR: .env 에 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 설정 필요")
        print("발급: https://developers.naver.com/apps/")
        return

    all_results = []

    if args.keywords:
        results = collect_for_category(args.category, args.keywords, args.per_keyword)
        all_results.extend(results)
    else:
        print(f"네이버 검색 API로 진짜 게시글 수집 시작")
        print(f"카테고리 {len(KEYWORDS_BY_CATEGORY)}개 × 키워드 × 3개 소스(카페/블로그/지식인)")
        print()

        for category, keywords in KEYWORDS_BY_CATEGORY.items():
            print(f"[{category}]")
            results = collect_for_category(category, keywords, args.per_keyword)
            all_results.extend(results)
            print(f"  카테고리 {category}: 누적 {len(all_results)}건")
            print()

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n총 {len(all_results)}건 수집")
    print(f"저장: {OUTPUT_PATH}")

    # 소스별 카운트
    from collections import Counter
    counts = Counter(r["source"] for r in all_results)
    print("\n[소스별 수집량]")
    for src, n in counts.most_common():
        print(f"  {src:<20} {n}건")

    # 샘플
    if all_results:
        print("\n[샘플 3건]")
        for r in all_results[:3]:
            print(f"  [{r['source']}] {r['title'][:50]}")
            print(f"    {r['content'][:100]}")
            print()


if __name__ == "__main__":
    main()
