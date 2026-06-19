"""
플랫폼 가중 페르소나 생성기

기존 PopulationGenerator를 활용하되,
플랫폼 인구 분포 + 플랫폼 특성을 덮어쓴다.
"""

import random
import uuid
from typing import Optional

from ..models.persona import Persona, PersonaAttributes, AgeGroup, Gender
from .platform_data import SNSPlatform, get_profile, PlatformProfile


class PlatformPersonaGenerator:
    """플랫폼 사용자 특성을 반영한 페르소나 생성"""

    def __init__(self, platform: SNSPlatform | str):
        self.platform = SNSPlatform(platform) if isinstance(platform, str) else platform
        self.profile: PlatformProfile = get_profile(self.platform)

    # ─────────────────────────────────────────────────
    # 샘플링 헬퍼
    # ─────────────────────────────────────────────────

    def _sample_age_group(self) -> AgeGroup:
        items = list(self.profile.age_distribution.items())
        labels = [a for a, _ in items]
        weights = [w for _, w in items]
        chosen = random.choices(labels, weights=weights, k=1)[0]
        return AgeGroup(chosen)

    def _sample_gender(self) -> Gender:
        ratio = self.profile.gender_ratio
        return Gender("male") if random.random() < ratio.get("male", 0.5) else Gender("female")

    # 연령대별 부적합 직업 (제외 키워드)
    _AGE_EXCLUDE = {
        AgeGroup.TEENS: ["직장인", "임원", "은퇴", "자영업", "대학원생",
                          "주부", "전업맘", "프리랜서", "콘텐츠", "워킹맘",
                          "헬스 트레이너", "요가 강사", "쇼핑몰 운영자",
                          "스타트업 대표", "디자이너"],
        AgeGroup.TWENTIES: ["중학생", "고등학생", "은퇴", "임원", "주부",
                            "전업맘", "직장인(과장)", "직장인(부장)",
                            "스타트업 대표", "워킹맘"],
        AgeGroup.THIRTIES: ["중학생", "고등학생", "은퇴", "임원", "직장인(부장)"],
        AgeGroup.FORTIES: ["중학생", "고등학생", "대학생", "은퇴",
                           "직장인(신입)", "직장인(대리)"],
        AgeGroup.FIFTIES: ["중학생", "고등학생", "대학생", "대학원생",
                           "직장인(신입)", "직장인(대리)", "워킹맘"],
        AgeGroup.SIXTIES_PLUS: ["중학생", "고등학생", "대학생", "대학원생",
                                "직장인(신입)", "직장인(대리)", "직장인(과장)",
                                "직장인(부장)", "콘텐츠", "IT", "워킹맘",
                                "트레이너", "강사", "디자이너", "스타트업"],
    }

    # 성별 제약 (해당 성별만 가능한 직업 키워드)
    _FEMALE_ONLY = ["주부", "전업맘", "워킹맘"]
    _MALE_STRONG = ["군인"]  # 남성 90% 확률

    def _sample_occupation(self, age_group: AgeGroup, gender: Gender) -> str:
        """연령대 + 성별 + 플랫폼 주요 직업 교차"""
        exclude = self._AGE_EXCLUDE.get(age_group, [])

        candidates = [
            j for j in self.profile.dominant_occupations
            if not any(ex in j for ex in exclude)
        ]

        # 성별 필터링
        if gender == Gender.MALE:
            candidates = [j for j in candidates
                          if not any(f in j for f in self._FEMALE_ONLY)]
        else:  # FEMALE
            # 군인 제외 (대부분 남성)
            candidates = [j for j in candidates
                          if not any(m in j for m in self._MALE_STRONG)]

        # 후보가 비면 연령대 기본 직업으로 폴백
        if not candidates:
            if age_group == AgeGroup.TEENS:
                candidates = ["중학생", "고등학생"]
            elif age_group == AgeGroup.TWENTIES:
                candidates = ["대학생", "직장인(신입)", "프리랜서"]
            elif age_group == AgeGroup.THIRTIES:
                candidates = ["직장인(대리)", "프리랜서", "자영업"]
            elif age_group == AgeGroup.FORTIES:
                candidates = ["직장인(과장)", "자영업"] + (["주부"] if gender == Gender.FEMALE else [])
            elif age_group == AgeGroup.FIFTIES:
                candidates = ["직장인(부장)", "자영업"] + (["주부"] if gender == Gender.FEMALE else [])
            else:  # 60대+
                candidates = ["은퇴", "자영업(은퇴예정)"] + (["주부"] if gender == Gender.FEMALE else [])

        return random.choice(candidates)

    def _sample_traits(self, base_count: int = 2) -> list[str]:
        """
        기본 특성 base_count개 + 플랫폼 특성 2~3개
        합쳐서 5개 이내
        """
        # 일반 특성 풀에서 base_count개 (다양성 확보)
        general_traits = [
            "가성비 중시", "트렌드 민감", "건강 관심", "품질 중시",
            "실용적", "자유로움", "효율 중시", "감성적",
            "보수적", "프리미엄 선호", "워라밸", "자기계발",
        ]
        base = random.sample(general_traits, min(base_count, len(general_traits)))

        # 플랫폼 특성 2~3개 (반드시 포함)
        platform_n = random.choice([2, 3])
        platform_picked = random.sample(
            self.profile.platform_traits,
            min(platform_n, len(self.profile.platform_traits)),
        )

        # 합치고 5개 이내로 제한
        combined = base + platform_picked
        return combined[:5]

    # ─────────────────────────────────────────────────
    # 페르소나 생성
    # ─────────────────────────────────────────────────

    def generate_one(self, name: Optional[str] = None) -> Persona:
        age = self._sample_age_group()
        gender = self._sample_gender()
        occupation = self._sample_occupation(age, gender)
        traits = self._sample_traits()

        attrs = PersonaAttributes(
            age_group=age,
            gender=gender,
            occupation=occupation,
            traits=traits,
            interests=self.profile.common_topics[:3],
        )

        return Persona(
            id=str(uuid.uuid4()),
            name=name or self._make_name(gender),
            attributes=attrs,
            weight=1.0,
            backstory=None,
        )

    def generate_panel(self, count: int) -> list[Persona]:
        return [self.generate_one() for _ in range(count)]

    # ─────────────────────────────────────────────────
    # 이름 생성 (간단 버전)
    # ─────────────────────────────────────────────────

    _MALE_NAMES = ["민준", "서준", "예준", "도윤", "주원", "시우", "지호", "선우", "현우", "준서"]
    _FEMALE_NAMES = ["서연", "지우", "지유", "수아", "지아", "하윤", "유나", "서윤", "민서", "예린"]
    _SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]

    def _make_name(self, gender: Gender) -> str:
        surname = random.choice(self._SURNAMES)
        given = random.choice(self._MALE_NAMES if gender == Gender.MALE else self._FEMALE_NAMES)
        return surname + given
