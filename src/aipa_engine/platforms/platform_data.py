"""
SNS 플랫폼별 사용자 프로필 데이터

출처 (공개 보고서 기반):
  - KISA 인터넷 이용실태조사 2024
  - 메조미디어 트렌드 리포트 / 세대별 분석
  - 오픈서베이 SNS 트렌드 리포트
  - 한국언론진흥재단 미디어이용조사
  - 한국갤럽 SNS 이용률 조사
  - 학술 논문 (디시인사이드, 익명 커뮤니티 사용자 연구)

각 플랫폼 프로필은 다음을 포함:
  - 연령 분포 (가중치)
  - 성비
  - 주요 직업 후보
  - 사용자 특성 키워드 (가치관/행동/태도)
  - 반응 톤 가이드
  - 자주 다루는 주제
"""

from dataclasses import dataclass, field
from enum import Enum


class SNSPlatform(str, Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    NAVER = "naver"
    GOOGLE = "google"
    DAANGN = "daangn"
    DCINSIDE = "dcinside"


@dataclass
class PlatformProfile:
    name_kr: str
    description: str

    # 연령 분포 가중치 (합 1.0)
    age_distribution: dict[str, float] = field(default_factory=dict)

    # 성비 (male / female 합 1.0)
    gender_ratio: dict[str, float] = field(default_factory=dict)

    # 주요 직업군 (가중치 높은 순)
    dominant_occupations: list[str] = field(default_factory=list)

    # 플랫폼 고유 사용자 특성 (페르소나 traits에 추가됨)
    platform_traits: list[str] = field(default_factory=list)

    # 반응 톤 가이드 (0.5B 모델 프롬프트용)
    tone_guide: str = ""

    # 자주 다루는 주제
    common_topics: list[str] = field(default_factory=list)

    # 보고서 출처
    sources: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# 플랫폼 프로필 (공개 데이터 기반)
# ─────────────────────────────────────────────────────────────────

PLATFORM_PROFILES: dict[SNSPlatform, PlatformProfile] = {
    SNSPlatform.YOUTUBE: PlatformProfile(
        name_kr="유튜브",
        description="동영상 콘텐츠 중심의 글로벌 플랫폼",
        age_distribution={
            "10대": 0.18, "20대": 0.22, "30대": 0.20,
            "40대": 0.18, "50대": 0.13, "60대+": 0.09,
        },
        gender_ratio={"male": 0.52, "female": 0.48},
        dominant_occupations=[
            "직장인(신입)", "직장인(대리)", "대학생", "고등학생",
            "프리랜서", "주부", "콘텐츠 제작",
        ],
        platform_traits=[
            "알고리즘 의존", "장시간 시청", "콘텐츠 충성도",
            "댓글 토론", "구독 채널 위주", "리뷰 신뢰", "유튜브 시청",
        ],
        tone_guide="구체적이고 의견 명확. '~인 듯', '~같아요' 어미. 알고리즘/추천 언급 자연스러움.",
        common_topics=["엔터테인먼트", "정보", "음악", "리뷰", "교육", "게임"],
        sources=["KISA 2024", "메조미디어 유튜브 사용자 분석"],
    ),

    SNSPlatform.INSTAGRAM: PlatformProfile(
        name_kr="인스타그램",
        description="이미지/비디오 공유 SNS",
        age_distribution={
            "10대": 0.20, "20대": 0.38, "30대": 0.22,
            "40대": 0.12, "50대": 0.06, "60대+": 0.02,
        },
        gender_ratio={"male": 0.42, "female": 0.58},
        dominant_occupations=[
            "대학생", "직장인(신입)", "프리랜서 디자이너",
            "쇼핑몰 운영자", "요가 강사", "헬스 트레이너",
            "IT 워킹맘", "고등학생",
        ],
        platform_traits=[
            "비주얼 중심", "스토리 문화", "인플루언서 영향",
            "라이프스타일 민감", "비교 의식", "트렌드 민감", "인스타 활발",
            "뷰티 관심", "패션 관심",
        ],
        tone_guide="감성적 표현. 이모지 사용. '!!', '느낌이 좋아요' 같은 감탄. 비주얼/분위기 언급.",
        common_topics=["패션", "뷰티", "여행", "음식", "운동", "인테리어"],
        sources=["메조미디어 2024", "오픈서베이 인스타그램 사용자 분석"],
    ),

    SNSPlatform.TWITTER: PlatformProfile(
        name_kr="X (트위터)",
        description="실시간 소셜 미디어 플랫폼",
        age_distribution={
            "10대": 0.12, "20대": 0.40, "30대": 0.28,
            "40대": 0.13, "50대": 0.05, "60대+": 0.02,
        },
        gender_ratio={"male": 0.48, "female": 0.52},
        dominant_occupations=[
            "직장인(신입)", "직장인(대리)", "대학생",
            "프리랜서", "콘텐츠 제작", "쇼핑몰 운영자",
        ],
        platform_traits=[
            "실시간 이슈", "짧은 글", "트렌드 추종",
            "정치/사회 민감", "강한 의견", "팬덤 활동",
            "리트윗 문화", "이슈 즉각 반응",
        ],
        tone_guide="짧고 임팩트. 이슈 키워드 언급. '~함', '~하더라' 등 단정형. 종종 빈정거림.",
        common_topics=["이슈", "팬덤", "정치", "밈", "사회 비평", "K-pop"],
        sources=["한국언론진흥재단", "트위터 한국 사용자 분석"],
    ),

    SNSPlatform.TIKTOK: PlatformProfile(
        name_kr="틱톡",
        description="숏폼 영상 플랫폼",
        age_distribution={
            "10대": 0.35, "20대": 0.40, "30대": 0.15,
            "40대": 0.07, "50대": 0.02, "60대+": 0.01,
        },
        gender_ratio={"male": 0.40, "female": 0.60},
        dominant_occupations=[
            "중학생", "고등학생", "대학생", "직장인(신입)",
            "콘텐츠 제작", "파트타임 스트리머",
        ],
        platform_traits=[
            "짧은 집중", "트렌드 선도", "Z세대 코드",
            "비주얼 우선", "챌린지 참여", "틱톡 활발",
            "K-pop 팬", "유행어 즉각 흡수",
        ],
        tone_guide="짧고 캐주얼. 신조어/줄임말 활발. 'ㄹㅇ', 'ㄷㄷ' 같은 초성. 이모지 적극.",
        common_topics=["챌린지", "K-pop", "댄스", "유머", "메이크업", "푸드"],
        sources=["메조미디어 Z세대 리포트", "틱톡 코리아 트렌드"],
    ),

    SNSPlatform.NAVER: PlatformProfile(
        name_kr="네이버",
        description="국내 최대 검색 포털 및 커뮤니티",
        age_distribution={
            "10대": 0.05, "20대": 0.18, "30대": 0.25,
            "40대": 0.25, "50대": 0.18, "60대+": 0.09,
        },
        gender_ratio={"male": 0.48, "female": 0.52},
        dominant_occupations=[
            "주부", "직장인(대리)", "직장인(과장)", "자영업",
            "전업맘", "은퇴", "공무원", "IT 워킹맘",
        ],
        platform_traits=[
            "정보 추구", "후기 신뢰", "지역/관심 기반",
            "실용적", "카페 활동", "블로그 글 작성",
            "지식인 의존", "꼼꼼한 비교",
        ],
        tone_guide="정중한 존댓말. '~인 것 같습니다', '~하시는 게 좋아요' 같은 권유형. 후기 인용.",
        common_topics=["맘카페", "맛집", "여행 후기", "재테크", "육아", "건강 정보"],
        sources=["KISA 2024", "네이버 데이터랩"],
    ),

    SNSPlatform.GOOGLE: PlatformProfile(
        name_kr="구글",
        description="글로벌 검색 및 리뷰 플랫폼",
        age_distribution={
            "10대": 0.10, "20대": 0.30, "30대": 0.25,
            "40대": 0.18, "50대": 0.12, "60대+": 0.05,
        },
        gender_ratio={"male": 0.55, "female": 0.45},
        dominant_occupations=[
            "직장인(대리)", "프리랜서", "프리랜서 디자이너",
            "대학원생", "IT 워킹맘", "콘텐츠 제작",
            "스타트업 대표", "웹디자이너",
        ],
        platform_traits=[
            "글로벌 정보", "리뷰 활발", "영어 콘텐츠 친숙",
            "기술 친화", "전문 정보 추구", "객관적 검색",
            "맥북 유저", "디지털 친화",
        ],
        tone_guide="정보 중심. 영어 단어 자연 혼용. '실제 사용해보니', '결론적으로' 같은 분석형.",
        common_topics=["기술", "여행", "맛집 리뷰", "전문 정보", "글로벌 트렌드"],
        sources=["나스미디어 NPR 2024", "Google Trends"],
    ),

    SNSPlatform.DAANGN: PlatformProfile(
        name_kr="당근",
        description="지역 기반 중고거래 및 커뮤니티",
        age_distribution={
            "10대": 0.05, "20대": 0.22, "30대": 0.30,
            "40대": 0.25, "50대": 0.13, "60대+": 0.05,
        },
        gender_ratio={"male": 0.45, "female": 0.55},
        dominant_occupations=[
            "주부", "직장인(대리)", "전업맘", "자영업",
            "IT 워킹맘", "프리랜서", "은퇴",
        ],
        platform_traits=[
            "지역 커뮤니티", "중고거래", "동네 생활",
            "실용적", "가성비 중시", "이웃 신뢰",
            "절약", "알뜰 소비", "거래 중심",
        ],
        tone_guide="친근한 반말 또는 동네 사람 같은 톤. '~예요', '직거래' 등 거래 용어. 솔직.",
        common_topics=["중고거래", "동네 정보", "맘카페 이슈", "지역 맛집", "이사 후기"],
        sources=["당근마켓 공식 사용자 통계", "한국갤럽 SNS 이용 조사"],
    ),

    SNSPlatform.DCINSIDE: PlatformProfile(
        name_kr="디시인사이드",
        description="국내 대표 익명 커뮤니티 사이트",
        age_distribution={
            "10대": 0.08, "20대": 0.42, "30대": 0.32,
            "40대": 0.13, "50대": 0.04, "60대+": 0.01,
        },
        gender_ratio={"male": 0.78, "female": 0.22},
        dominant_occupations=[
            "대학생", "직장인(신입)", "프리랜서", "군인",
            "콘텐츠 제작", "고등학생", "자영업",
        ],
        platform_traits=[
            "익명적", "냉소적", "주류 반감",
            "갤러리 결속", "솔직함", "공격성",
            "유머/풍자", "정보 검증 의심", "은어/밈 활발",
        ],
        tone_guide="반말. 신조어/은어 활발. '~ㄴ뎁;', '~함', '~ㅇㅇ' 같은 짧은 어미. 빈정거림 자연.",
        common_topics=["갤러리별 주제", "이슈 비평", "밈", "관심사 심층 토론", "정치 풍자"],
        sources=[
            "박OO(2022) 「디시인사이드 사용자의 익명성과 자기표현」",
            "익명 커뮤니티 사용자 연구 학술 논문 종합",
        ],
    ),
}


def get_profile(platform: SNSPlatform | str) -> PlatformProfile:
    """플랫폼 enum 또는 문자열로 프로필 조회"""
    if isinstance(platform, str):
        platform = SNSPlatform(platform.lower())
    return PLATFORM_PROFILES[platform]


def list_platforms() -> list[dict]:
    """API 용 플랫폼 목록 (선택 화면용)"""
    return [
        {
            "id": p.value,
            "name": prof.name_kr,
            "description": prof.description,
            "tags": prof.platform_traits[:3],
        }
        for p, prof in PLATFORM_PROFILES.items()
    ]
