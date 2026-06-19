"""
유튜브 실제 댓글 수집 (API 키 불필요)

youtube-comment-downloader 라이브러리 사용:
  - YouTube 공개 댓글 페이지를 그대로 파싱
  - API 키 / 인증 불필요
  - 한국 인기 동영상에서 실제 한국어 댓글 수집 가능

수집 데이터:
  - 영상 ID, 댓글 본문, 좋아요 수, 작성 시간, 작성자
  - 카테고리(검색 키워드)별 분류

출력:
  data/platform_tone/youtube_real_comments.jsonl

사용법:
  python scripts/collect_youtube_comments.py
  python scripts/collect_youtube_comments.py --keywords "유니클로 신상" --per-video 50
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import urllib.parse
import requests
import re

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "platform_tone" / "youtube_real_comments.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# AIPA 학습 카테고리와 매칭되는 검색 키워드
KEYWORDS_BY_CATEGORY = {
    "식품": ["신상 치킨 리뷰", "프리미엄 도시락 후기"],
    "화장품": ["선크림 추천", "립틴트 추천"],
    "앱/서비스": ["가계부 앱 추천", "AI 영어 회화 앱"],
    "콘텐츠": ["넷플릭스 신작 리뷰", "OTT 추천"],
    "패션": ["유니클로 신상", "오버사이즈 티"],
    "가전/전자": ["무선이어폰 추천", "노이즈캔슬링 리뷰"],
    "자동차": ["아이오닉 6 리뷰", "전기차 추천"],
    "교육": ["코딩 부트캠프 후기", "온라인 강의 추천"],
}


def search_youtube_videos(query: str, max_results: int = 3) -> list[str]:
    """
    YouTube 검색 결과 페이지 HTML에서 영상 ID 추출
    (API 사용 안 함 - 공개 검색 페이지 파싱)
    """
    encoded_q = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={encoded_q}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []

        # videoId 추출 (HTML에 "videoId":"xxxxx" 형태로 들어있음)
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', r.text)
        # 중복 제거 + 순서 유지
        seen = set()
        unique_ids = []
        for vid in video_ids:
            if vid not in seen:
                seen.add(vid)
                unique_ids.append(vid)
            if len(unique_ids) >= max_results:
                break
        return unique_ids
    except Exception as e:
        print(f"    검색 실패: {e}")
        return []


def collect_comments_for_video(video_id: str, max_comments: int = 30) -> list[dict]:
    """1개 영상에서 인기 댓글 수집"""
    downloader = YoutubeCommentDownloader()
    url = f"https://www.youtube.com/watch?v={video_id}"

    comments = []
    try:
        for comment in downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
            text = comment.get("text", "").strip()
            if not text or len(text) < 5:
                continue

            comments.append({
                "video_id": video_id,
                "comment_id": comment.get("cid"),
                "text": text,
                "votes": comment.get("votes", "0"),
                "author": comment.get("author"),
                "time": comment.get("time"),
                "reply_count": comment.get("reply_count", 0),
            })

            if len(comments) >= max_comments:
                break
    except Exception as e:
        print(f"    댓글 수집 실패 ({video_id}): {e}")

    return comments


def collect_for_category(category: str, keywords: list[str],
                          videos_per_keyword: int, comments_per_video: int) -> list[dict]:
    """카테고리 하나에 대해 검색 + 댓글 수집"""
    collected = []

    for kw in keywords:
        print(f"  검색: '{kw}'")
        video_ids = search_youtube_videos(kw, max_results=videos_per_keyword)
        print(f"    영상 {len(video_ids)}개 찾음: {video_ids}")

        for vid in video_ids:
            print(f"    [{vid}] 댓글 수집 중...")
            comments = collect_comments_for_video(vid, max_comments=comments_per_video)
            print(f"      {len(comments)}건 수집")

            for c in comments:
                collected.append({
                    "source": "youtube",
                    "platform": "youtube",
                    "category": category,
                    "search_keyword": kw,
                    "video_id": c["video_id"],
                    "text": c["text"],
                    "votes": c["votes"],
                    "author": c["author"],
                    "time": c["time"],
                    "reply_count": c["reply_count"],
                    "collected_at": datetime.utcnow().isoformat(),
                })

            # 너무 빠른 요청 방지
            time.sleep(0.5)

    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-per-keyword", type=int, default=2, help="키워드당 검색할 영상 수")
    parser.add_argument("--comments-per-video", type=int, default=20, help="영상당 수집할 댓글 수")
    parser.add_argument("--keywords", nargs="*", default=None, help="직접 키워드 지정")
    parser.add_argument("--category", type=str, default="콘텐츠")
    args = parser.parse_args()

    all_results = []

    if args.keywords:
        print(f"키워드 직접 지정 모드: {args.keywords}")
        all_results = collect_for_category(
            args.category, args.keywords,
            args.videos_per_keyword, args.comments_per_video,
        )
    else:
        print(f"유튜브 실제 댓글 수집 시작")
        print(f"카테고리 {len(KEYWORDS_BY_CATEGORY)}개 × 키워드 × 영상 {args.videos_per_keyword}개 × 댓글 {args.comments_per_video}개")
        print()

        for category, keywords in KEYWORDS_BY_CATEGORY.items():
            print(f"[{category}]")
            results = collect_for_category(
                category, keywords,
                args.videos_per_keyword, args.comments_per_video,
            )
            all_results.extend(results)
            print(f"  카테고리 {category}: 누적 {len(all_results)}건")
            print()

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n총 {len(all_results)}건 수집")
    print(f"저장: {OUTPUT_PATH}")

    # 카테고리별 카운트
    from collections import Counter
    counts = Counter(r["category"] for r in all_results)
    print("\n[카테고리별 수집량]")
    for cat, n in counts.most_common():
        print(f"  {cat:<12} {n}건")

    # 샘플
    if all_results:
        print("\n[샘플 댓글 5건]")
        for r in all_results[:5]:
            print(f"  [{r['category']}] {r['text'][:100]}")


if __name__ == "__main__":
    main()
