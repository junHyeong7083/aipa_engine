"""
캘리브레이션 서비스 (Calibration Service)

C#으로 비유하면 통계 보정 알고리즘을 담당하는 서비스 클래스.

핵심 알고리즘: IPF (Iterative Proportional Fitting) / Raking
= 설문 응답의 가중치를 조정해서 목표 인구통계 분포에 맞추는 기법

예를 들어 10명 중 남자 8명이 나왔는데 실제 인구는 5:5이면,
남자 응답에 낮은 가중치, 여자 응답에 높은 가중치를 부여해서 보정.
실제 여론조사/마케팅 리서치에서 필수적으로 사용하는 기법임.
"""

from typing import Optional
# numpy = 수학/행렬 연산 라이브러리 (C#의 MathNet.Numerics)
import numpy as np

from ..models.persona import Persona, PersonaConfig, AgeGroup, Gender
from ..models.survey import SurveyResponse


class Calibrator:
    """
    설문 응답 캘리브레이터 (C#의 public class Calibrator : ICalibrator)

    IPF/Raking 알고리즘으로 가중치를 조정해서
    샘플 분포가 목표 인구통계 분포와 일치하도록 보정.
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        max_iterations: 최대 반복 횟수 (수렴 안 하면 여기서 멈춤)
        tolerance: 수렴 판정 기준 (가중치 변화가 이것보다 작으면 완료)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    async def calibrate(
        self,
        personas: list[Persona],
        responses: list[SurveyResponse],
        config: PersonaConfig,
        target_marginals: Optional[dict] = None,
    ) -> list[SurveyResponse]:
        """
        메인 캘리브레이션 실행 (C#의 public async Task<List<SurveyResponse>> CalibrateAsync(...))

        1. 목표 분포 설정 (타겟 인구통계)
        2. IPF/Raking으로 가중치 계산
        3. 각 페르소나/응답에 가중치 적용

        예: 남자가 과대표집됐으면 남자 응답의 가중치를 낮추고 여자를 높임
        """
        if not personas or not responses:
            return responses

        # 목표 한계분포 생성 (유저가 직접 지정 안 했으면 config에서 자동 생성)
        if target_marginals is None:
            target_marginals = self._build_target_marginals(config)

        # 페르소나 ID → 객체 매핑 (빠른 조회용)
        # C#의 personas.ToDictionary(p => p.Id)
        persona_map = {p.id: p for p in personas}

        # 초기 가중치 = 전부 1.0 (보정 전)
        weights = np.ones(len(personas))

        # IPF/Raking 알고리즘 실행 → 보정된 가중치 반환
        weights = self._run_raking(personas, weights, target_marginals)

        # 계산된 가중치를 페르소나에 적용
        for i, persona in enumerate(personas):
            persona.weight = float(weights[i])

        # 각 응답의 가중치도 해당 페르소나의 가중치로 업데이트
        for response in responses:
            persona = persona_map.get(response.persona_id)
            if persona:
                response.weight = persona.weight

        return responses

    def _build_target_marginals(self, config: PersonaConfig) -> dict:
        """
        목표 한계분포(Target Marginals) 생성

        C#의 Dictionary<string, Dictionary<string, double>> 같은 것.
        예: {"age": {"20대": 0.5, "30대": 0.5}, "gender": {"male": 0.5, "female": 0.5}}
        """
        marginals = {}

        # 연령 분포
        if config.age_groups:
            # 유저가 지정한 연령대 → 균등 분배
            prob = 1.0 / len(config.age_groups)
            marginals["age"] = {ag.value: prob for ag in config.age_groups}
        else:
            # 실제 한국 인구 통계 비율
            marginals["age"] = {
                "10대": 0.09, "20대": 0.13, "30대": 0.14,
                "40대": 0.16, "50대": 0.17, "60대+": 0.31,
            }

        # 성별 분포
        marginals["gender"] = config.gender_ratio

        return marginals

    def _run_raking(
        self,
        personas: list[Persona],
        initial_weights: np.ndarray,
        target_marginals: dict,
    ) -> np.ndarray:
        """
        Raking(IPF) 알고리즘 실행 (핵심 알고리즘)

        C#으로 비유하면 for 루프 안에서 가중치를 반복 조정하는 것.

        원리:
        1. 각 변수(연령, 성별)에 대해 순서대로 가중치 조정
        2. 조정된 가중치가 안정될 때까지 반복
        3. 수렴하면 멈춤

        예: 1회차에서 연령 보정 → 2회차에서 성별 보정 → 3회차에서 연령 다시 보정 → ...
        → 연령도 맞고 성별도 맞는 가중치가 나옴
        """
        weights = initial_weights.copy()
        n = len(personas)

        for iteration in range(self.max_iterations):
            prev_weights = weights.copy()  # 이전 가중치 저장 (수렴 판정용)

            # 각 변수(연령, 성별 등)에 대해 순서대로 조정
            for variable, target_dist in target_marginals.items():
                weights = self._adjust_for_marginal(
                    personas, weights, variable, target_dist
                )

            # 수렴 확인: 가중치 변화가 tolerance 미만이면 완료
            max_change = np.max(np.abs(weights - prev_weights))
            if max_change < self.tolerance:
                break  # 수렴 완료

        # 가중치 정규화: 합이 원래 샘플 크기(n)와 같도록 조정
        # → 평균 가중치가 1.0이 되게 함
        weights = weights * n / weights.sum()

        return weights

    def _adjust_for_marginal(
        self,
        personas: list[Persona],
        weights: np.ndarray,
        variable: str,
        target_dist: dict,
    ) -> np.ndarray:
        """
        단일 변수에 대한 가중치 조정

        예: 연령 변수에서 "20대" 목표가 50%인데 현재 30%이면
        → 20대 페르소나들의 가중치를 50/30 = 1.67배로 늘림
        """
        new_weights = weights.copy()

        for category, target_prop in target_dist.items():
            # 이 카테고리에 해당하는 페르소나 찾기 (boolean mask)
            mask = self._get_category_mask(personas, variable, category)

            if not np.any(mask):  # 해당 카테고리에 아무도 없으면 스킵
                continue

            # 현재 가중 비율 계산
            current_prop = weights[mask].sum() / weights.sum()

            if current_prop > 0:
                # 조정 비율 = 목표/현재 (예: 0.5/0.3 = 1.67)
                factor = target_prop / current_prop
                new_weights[mask] *= factor  # 해당 그룹 가중치에 비율 곱하기

        return new_weights

    def _get_category_mask(
        self,
        personas: list[Persona],
        variable: str,
        category: str,
    ) -> np.ndarray:
        """
        특정 카테고리에 해당하는 페르소나 boolean 마스크 생성

        C#의 personas.Select(p => p.AgeGroup == category).ToArray() 같은 것.
        True/False 배열을 반환 → numpy 인덱싱에 사용
        """
        mask = np.zeros(len(personas), dtype=bool)

        for i, persona in enumerate(personas):
            if variable == "age":
                mask[i] = persona.attributes.age_group.value == category
            elif variable == "gender":
                gender_str = "male" if persona.attributes.gender == Gender.MALE else "female"
                mask[i] = gender_str == category
            elif variable == "occupation":
                mask[i] = persona.attributes.occupation == category

        return mask

    def calculate_distribution_fidelity(
        self,
        personas: list[Persona],
        target_marginals: dict,
    ) -> float:
        """
        분포 충실도(Fidelity) 계산 - 샘플이 목표 분포와 얼마나 일치하는지

        0.0 = 완전 불일치, 1.0 = 완벽 일치
        모든 카테고리의 |실제비율 - 목표비율| 평균을 1에서 빼서 계산

        C#의 public double CalculateFidelity(List<Persona> personas, ...) 같은 것
        """
        if not personas:
            return 0.0

        total_error = 0.0
        n_categories = 0

        weights = np.array([p.weight for p in personas])
        total_weight = weights.sum()

        for variable, target_dist in target_marginals.items():
            for category, target_prop in target_dist.items():
                mask = self._get_category_mask(personas, variable, category)
                # 실제 가중 비율 계산
                actual_prop = weights[mask].sum() / total_weight if total_weight > 0 else 0

                # 오차 = |실제 - 목표|
                error = abs(actual_prop - target_prop)
                total_error += error
                n_categories += 1

        if n_categories == 0:
            return 1.0

        # 평균 오차를 1에서 빼서 충실도 점수로 변환
        avg_error = total_error / n_categories
        fidelity = 1.0 - min(avg_error, 1.0)

        return fidelity
