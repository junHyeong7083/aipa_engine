"""
디시인사이드 실제 게시글 수집

주의:
  - 학술/연구/개인 분석 목적으로만 사용
  - robots.txt 및 약관 확인 필요
  - 너무 빠른 요청 자제 (sleep 적용)
  - 민감 갤러리(정치, 혐오) 제외 권장

수집 대상:
  - 일반적인 관심사 갤러리(헬스, 컴퓨터, 음식 등)의 인기 게시글 제목
  - 게시글 본문 + 일부 댓글 (선택)

출력:
  data/platform_tone/dcinside_real_posts.jsonl

사용법:
  python scripts/collect_dcinside_posts.py
  python scripts/collect_dcinside_posts.py --galleries health computer
"""

import sys
import json
import time
import re
import argparse
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "platform_tone" / "dcinside_real_posts.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# AIPA 학습 카테고리와 매칭되는 디시 갤러리 (관심사 기반, 정치 제외)
GALLERY_BY_CATEGORY = {
    "가전/전자": ["dcbest", "comp_new1"],   # 컴퓨터본체 갤러리
    "콘텐츠": ["movie", "drama_new1"],       # 영화/드라마 갤
    "패션": ["fashion"],                      # 패션 갤
    "식품": ["cook", "food"],                # 요리/음식 갤
    "자동차": ["car_new1"],                  # 자동차 갤
    "교육": ["college", "publicedu"],        # 대학생/공무원 갤
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://gall.dcinside.com/",
}

LIST_URL = "https://gall.dcinside.com/board/lists/"
VIEW_URL = "https://gall.dcinside.com/board/view/"


def fetch_gallery_list(gallery_id: str, page: int = 1) -> list[dict]:
    """갤러리 목록 페이지에서 게시글 메타데이터 추출"""
    params = {"id": gallery_id, "page": page}
    try:
        r = requests.get(LIST_URL, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  [{gallery_id}] HTTP {r.status_code}")
            return []

        soup = BeautifulSoup(r.text, "lxml")

        # tr 단위로 게시글 행 파싱
        rows = soup.select("tr.us-post")
        posts = []
        for row in rows:
            # 공지/이벤트 제외
            cls = row.get("class", [])
            if any(c in cls for c in ["notice", "ad_notice"]):
                continue

            title_a = row.select_one(".gall_tit a")
            if not title_a:
                continue

            href = title_a.get("href", "")
            no_match = re.search(r"no=(\d+)", href)
            if not no_match:
                continue

            post_no = no_match.group(1)
            title = title_a.get_text(strip=True)

            views_el = row.select_one(".gall_count")
            recommend_el = row.select_one(".gall_recommend")

            posts.append({
                "no": post_no,
                "title": title,
                "views": views_el.get_text(strip=True) if views_el else "0",
                "recommend": recommend_el.get_text(strip=True) if recommend_el else "0",
            })

        return posts
    except Exception as e:
        print(f"  [{gallery_id}] 목록 실패: {e}")
        return []


def fetch_post_content(gallery_id: str, post_no: str) -> dict | None:
    """게시글 본문 + 일부 메타데이터 추출"""
    params = {"id": gallery_id, "no": post_no}
    try:
        r = requests.get(VIEW_URL, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "lxml")

        # 본문
        content_div = soup.select_one(".write_div")
        if not content_div:
            return None

        # 이미지 src 등 제거 후 텍스트만
        for img in content_div.find_all("img"):
            img.decompose()
        for script in content_div.find_all("script"):
            script.decompose()

        content = content_div.get_text("\n", strip=True)

        # 추천/비추천
        recommend = soup.select_one(".up_num")
        decline = soup.select_one(".down_num")

        return {
            "content": content,
            "recommend": recommend.get_text(strip=True) if recommend else "0",
            "decline": decline.get_text(strip=True) if decline else "0",
        }
    except Exception as e:
        return None


def collect_from_gallery(gallery_id: str, category: str,
                          max_posts: int = 20, fetch_content: bool = True,
                          max_pages: int = 1) -> list[dict]:
    """1개 갤러리에서 여러 페이지 순회하며 게시글 수집"""
    collected = []
    seen_nos: set[str] = set()

    for page in range(1, max_pages + 1):
        if len(collected) >= max_posts:
            break

        print(f"  [{gallery_id}] page {page} 목록 가져오는 중...")
        posts = fetch_gallery_list(gallery_id, page=page)
        if not posts:
            print(f"    페이지 {page} 비어있음 또는 차단")
            break
        print(f"    {len(posts)}개 게시글 발견")

        for post in posts:
            if len(collected) >= max_posts:
                break
            if post["no"] in seen_nos:
                continue
            seen_nos.add(post["no"])

            record = {
                "source": "dcinside",
                "platform": "dcinside",
                "category": category,
                "gallery": gallery_id,
                "post_no": post["no"],
                "title": post["title"],
                "views": post["views"],
                "recommend_list": post["recommend"],
                "page": page,
                "collected_at": datetime.utcnow().isoformat(),
            }

            if fetch_content:
                time.sleep(0.8)
                details = fetch_post_content(gallery_id, post["no"])
                if details:
                    record.update({
                        "content": details["content"][:1000],
                        "recommend": details["recommend"],
                        "decline": details["decline"],
                    })

            collected.append(record)

        time.sleep(1.0)  # 페이지 간 대기

    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-posts", type=int, default=10, help="갤러리당 최대 게시글 수")
    parser.add_argument("--max-pages", type=int, default=1, help="갤러리당 순회할 페이지 수")
    parser.add_argument("--galleries", nargs="*", default=None, help="특정 갤러리 ID 지정")
    parser.add_argument("--category", type=str, default="기타")
    parser.add_argument("--no-content", action="store_true", help="본문 안 가져오기 (제목만)")
    args = parser.parse_args()

    all_results = []
    fetch_content = not args.no_content

    if args.galleries:
        print(f"갤러리 직접 지정: {args.galleries}")
        for gid in args.galleries:
            results = collect_from_gallery(gid, args.category, args.max_posts, fetch_content, args.max_pages)
            all_results.extend(results)
    else:
        print("디시인사이드 실제 게시글 수집 시작")
        print(f"카테고리 {len(GALLERY_BY_CATEGORY)}개")
        print()

        for category, galleries in GALLERY_BY_CATEGORY.items():
            print(f"[{category}]")
            for gid in galleries:
                results = collect_from_gallery(gid, category, args.max_posts, fetch_content, args.max_pages)
                all_results.extend(results)
                print(f"  누적: {len(all_results)}건")
                time.sleep(1.0)  # 갤러리 간 대기
            print()

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n총 {len(all_results)}건 수집")
    print(f"저장: {OUTPUT_PATH}")

    # 갤러리별 카운트
    from collections import Counter
    counts = Counter(r["gallery"] for r in all_results)
    print("\n[갤러리별 수집량]")
    for gid, n in counts.most_common():
        print(f"  {gid:<15} {n}건")

    # 샘플
    if all_results:
        print("\n[샘플 게시글 5건]")
        for r in all_results[:5]:
            print(f"  [{r['gallery']}] {r['title'][:80]}")
            content = r.get("content", "")
            if content:
                print(f"    {content[:100]}")


if __name__ == "__main__":
    main()
