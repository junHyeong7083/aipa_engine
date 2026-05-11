"""
인구 합성 생성기 (Synthetic Population Generator)

C#으로 비유하면 통계 기반으로 가상 인물을 생성하는 Factory 패턴 서비스.
KOSIS(통계청) 인구 분포 데이터를 기반으로 현실적인 페르소나 패널을 만듦.

핵심 알고리즘: IPF (Iterative Proportional Fitting)
= 주어진 통계 분포에 맞게 가중치를 조정하는 알고리즘
"""

import random
# uuid4 = C#의 Guid.NewGuid()
from uuid import uuid4
from typing import Any, Optional

# numpy = C#의 MathNet.Numerics 같은 수학/통계 라이브러리
import numpy as np

# 데이터 모델 import
from ..models.persona import (
    Persona,
    PersonaAttributes,
    PersonaConfig,
    Gender,
    AgeGroup,
)


# === 한국 이름 데이터 (페르소나 이름 생성용) ===
KOREAN_SURNAMES = [
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍",
]
KOREAN_MALE_NAMES = [
    "민준", "서준", "예준", "도윤", "시우", "주원", "하준", "지호", "준우", "준서",
    "현우", "지훈", "건우", "우진", "선우", "태민", "승현", "재윤", "정우", "민혁",
]
KOREAN_FEMALE_NAMES = [
    "서연", "서윤", "지우", "서현", "민서", "하은", "하린", "윤서", "지민", "수아",
    "예은", "소율", "채원", "다은", "유진", "예진", "수빈", "지현", "은서", "시은",
]

# 기본 직업 목록
DEFAULT_OCCUPATIONS = [
    "회사원", "공무원", "자영업자", "프리랜서", "학생",
    "주부", "전문직", "서비스직", "생산직", "무직",
    "교사", "의료직", "IT 개발자", "금융업", "연구원",
]

# 기본 성격 특성 목록
DEFAULT_TRAITS = [
    "실용적", "트렌디", "보수적", "진보적", "가성비 중시",
    "품질 중시", "브랜드 선호", "친환경 선호", "디지털 친숙", "오프라인 선호",
    "건강 관심", "안전 중시", "편의 추구", "럭셔리 선호", "미니멀리스트",
    "자연주의", "얼리어답터", "가격 민감", "운동 매니아", "다이어터",
]

# 연령대별 특성 풀 (연령에 맞는 특성만 부여)
AGE_TRAIT_POOL = {
    AgeGroup.TEENS: ["트렌디", "가성비 중시", "디지털 친숙", "SNS 활발", "유행 민감", "또래 의식", "얼리어답터", "운동 매니아"],
    AgeGroup.TWENTIES: ["트렌디", "가성비 중시", "디지털 친숙", "SNS 활발", "자기계발", "소확행", "얼리어답터", "운동 매니아", "미니멀리스트", "친환경 선호", "브랜드 선호", "가격 민감"],
    AgeGroup.THIRTIES: ["실용적", "트렌디", "워라밸", "브랜드 선호", "건강 관심", "디지털 친숙", "품질 중시", "가성비 중시", "친환경 선호", "미니멀리스트", "자기계발"],
    AgeGroup.FORTIES: ["실용적", "안전 중시", "가족 중심", "품질 중시", "건강 관심", "경험 중시", "교육열", "브랜드 선호", "안정 추구"],
    AgeGroup.FIFTIES: ["건강 관심", "품질 중시", "보수적", "안전 중시", "실용적", "가족 중심", "안정 추구", "전통 선호", "알뜰 소비"],
    AgeGroup.SIXTIES_PLUS: ["건강 최우선", "보수적", "전통 선호", "안전 중시", "절약", "가족 중심", "오프라인 선호", "실용적"],
}


class PopulationGenerator:
    """
    통계 기반 가상 인구 생성기 (C#의 public class PopulationGenerator : IPopulationGenerator)

    실제 한국 인구통계(KOSIS)를 기반으로 현실적인 페르소나 패널을 생성.
    연령/성별/직업 분포가 실제 통계와 일치하도록 확률적 샘플링을 수행.
    """

    def __init__(self):
        # === 한계분포(Marginal Distribution) 초기화 ===
        # KOSIS 2024년 데이터 기준 실제 한국 인구 비율

        # 연령대별 인구 비율 (합계 = 1.0)
        self.age_distribution = {
            AgeGroup.TEENS: 0.09,           # 10대 9%
            AgeGroup.TWENTIES: 0.13,        # 20대 13%
            AgeGroup.THIRTIES: 0.14,        # 30대 14%
            AgeGroup.FORTIES: 0.16,         # 40대 16%
            AgeGroup.FIFTIES: 0.17,         # 50대 17%
            AgeGroup.SIXTIES_PLUS: 0.31,    # 60대+ 31%
        }

        # 성별 비율
        self.gender_distribution = {
            Gender.MALE: 0.499,     # 남성 49.9%
            Gender.FEMALE: 0.501,   # 여성 50.1%
        }

        # 직업별 비율
        self.occupation_distribution = {
            "회사원": 0.25, "공무원": 0.08, "자영업자": 0.10, "프리랜서": 0.05,
            "학생": 0.12, "주부": 0.08, "전문직": 0.12, "서비스직": 0.10,
            "생산직": 0.07, "무직": 0.03,
        }

        # 조건부 확률 테이블 설정 (연령대별 직업 분포)
        self._setup_conditional_distributions()

    def _setup_conditional_distributions(self):
        """
        조건부 확률 테이블 P(직업 | 연령대) 설정

        C#의 Dictionary<AgeGroup, Dictionary<string, double>> 같은 것.
        예: 10대 → 학생 85%, 무직 10%, 기타 5%
        실제로는 더 정교해야 하지만, 일단 간소화된 버전
        """
        self.occupation_given_age = {
            AgeGroup.TEENS: {"학생": 0.85, "무직": 0.10, "기타": 0.05},
            AgeGroup.TWENTIES: {"학생": 0.35, "회사원": 0.30, "프리랜서": 0.10, "기타": 0.25},
            AgeGroup.THIRTIES: {"회사원": 0.40, "자영업자": 0.15, "전문직": 0.15, "기타": 0.30},
            AgeGroup.FORTIES: {"회사원": 0.35, "자영업자": 0.20, "전문직": 0.15, "기타": 0.30},
            AgeGroup.FIFTIES: {"회사원": 0.25, "자영업자": 0.25, "주부": 0.15, "기타": 0.35},
            AgeGroup.SIXTIES_PLUS: {"무직": 0.35, "자영업자": 0.20, "주부": 0.20, "기타": 0.25},
        }

    async def generate(self, config: PersonaConfig) -> list[Persona]:
        """
        설정에 맞는 페르소나 패널 생성 (메인 메서드)
        C#의 public async Task<List<Persona>> GenerateAsync(PersonaConfig config)

        1. 연령/성별/직업 분포 결정
        2. 각 분포에서 확률적 샘플링
        3. 이름 생성 + 특성 부여
        """
        personas = []

        # 타겟 분포 계산 (유저가 지정한 게 있으면 그걸 사용, 없으면 실제 통계)
        age_dist = self._get_age_distribution(config)
        gender_dist = self._get_gender_distribution(config)
        occupation_pool = config.occupations if config.occupations else list(self.occupation_distribution.keys())
        trait_pool = config.traits if config.traits else DEFAULT_TRAITS

        # panel_count명 만큼 반복 생성
        for i in range(config.panel_count):
            # 확률 분포에서 랜덤 샘플링 (주사위 던지기 같은 것)
            age_group = self._sample_from_distribution(age_dist)        # 연령대 뽑기
            gender = self._sample_from_distribution(gender_dist)        # 성별 뽑기
            occupation = self._sample_occupation(age_group, occupation_pool)  # 직업 뽑기 (연령대 고려)
            # 연령대별 적절한 특성 풀에서 선택
            age_traits = AGE_TRAIT_POOL.get(age_group, DEFAULT_TRAITS)
            effective_trait_pool = config.traits if config.traits else age_traits
            traits = self._sample_traits(effective_trait_pool, num_traits=random.randint(2, 4))

            # 한국 이름 생성 (성별에 맞게)
            name = self._generate_korean_name(gender)

            # PersonaAttributes 객체 조립 (C#의 new PersonaAttributes { ... })
            attributes = PersonaAttributes(
                age_group=age_group,
                gender=gender,
                occupation=occupation,
                traits=traits,
            )

            # Persona 객체 생성
            persona = Persona(
                id=str(uuid4()),        # 고유 ID 생성
                name=name,
                attributes=attributes,
                weight=1.0,             # 초기 가중치 (캘리브레이션 전이라 1.0)
            )

            personas.append(persona)

        return personas

    def _get_age_distribution(self, config: PersonaConfig) -> dict[AgeGroup, float]:
        """
        연령대 분포 결정

        유저가 특정 연령대를 지정하면 → 균등 분배 (예: [20대, 30대] → 각 50%)
        지정 안 하면 → 실제 한국 인구 통계 비율 사용
        """
        if config.age_groups:
            prob = 1.0 / len(config.age_groups)  # 균등 분배
            return {ag: prob for ag in config.age_groups}
        return self.age_distribution

    def _get_gender_distribution(self, config: PersonaConfig) -> dict[Gender, float]:
        """성별 분포 결정 (유저 지정 성비 사용)"""
        return {
            Gender.MALE: config.gender_ratio.get("male", 0.5),
            Gender.FEMALE: config.gender_ratio.get("female", 0.5),
        }

    def _sample_from_distribution(self, distribution: dict) -> Any:
        """
        확률 분포에서 하나 샘플링 (가중 랜덤 선택)

        C#의 Random.NextDouble() + 누적확률 비교와 비슷한 개념.
        예: {"20대": 0.5, "30대": 0.5} → 50% 확률로 둘 중 하나 선택

        numpy.random.choice 사용 (인덱스 기반으로 원본 타입 보존)
        """
        items = list(distribution.keys())       # 선택지 목록
        probs = list(distribution.values())     # 각 선택지의 확률

        # 확률 합계를 1.0으로 정규화 (혹시 합이 1이 아닌 경우 대비)
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        # 인덱스로 샘플링 (numpy가 문자열을 바꿔버리는 문제 방지)
        idx = np.random.choice(len(items), p=probs)
        return items[idx]

    def _sample_occupation(self, age_group: AgeGroup, occupation_pool: list[str]) -> str:
        """
        직업 샘플링 (연령대를 고려한 조건부 확률 사용)

        예: 10대이면 학생이 85% 확률, 60대+이면 무직이 35% 확률
        → 단순 랜덤이 아니라 현실적인 조합이 나옴
        """
        # 해당 연령대의 조건부 직업 분포 가져오기
        cond_dist = self.occupation_given_age.get(age_group, {})

        # 유저가 지정한 직업 풀에서만 선택 가능하도록 필터링
        available = {}
        for occ in occupation_pool:
            if occ in cond_dist:
                available[occ] = cond_dist[occ]                     # 조건부 확률 있으면 사용
            elif occ in self.occupation_distribution:
                available[occ] = self.occupation_distribution[occ]  # 없으면 전체 분포에서 가져옴

        if not available:
            return random.choice(occupation_pool)  # 아무것도 매칭 안 되면 완전 랜덤

        return self._sample_from_distribution(available)

    def _sample_traits(self, trait_pool: list[str], num_traits: int) -> list[str]:
        """특성 랜덤 선택 (C#의 list.OrderBy(x => rng.Next()).Take(n) 같은 것)"""
        return random.sample(trait_pool, min(num_traits, len(trait_pool)))

    def _generate_korean_name(self, gender: Gender) -> str:
        """한국 이름 랜덤 생성 (성 + 이름 조합)"""
        surname = random.choice(KOREAN_SURNAMES)        # 랜덤 성씨
        if gender == Gender.MALE:
            given_name = random.choice(KOREAN_MALE_NAMES)   # 남자 이름
        else:
            given_name = random.choice(KOREAN_FEMALE_NAMES)  # 여자 이름
        return surname + given_name  # 예: "김민준"
