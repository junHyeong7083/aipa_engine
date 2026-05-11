"""Tests for Calibrator service (IPF/Raking algorithm)."""

import pytest
import numpy as np

from aipa_engine.models.persona import (
    Persona,
    PersonaAttributes,
    PersonaConfig,
    AgeGroup,
    Gender,
)
from aipa_engine.models.survey import SurveyResponse
from aipa_engine.services.calibrator import Calibrator


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_persona(pid: str, age: AgeGroup, gender: Gender) -> Persona:
    return Persona(
        id=pid,
        name=f"Test-{pid}",
        attributes=PersonaAttributes(
            age_group=age,
            gender=gender,
            occupation="회사원",
        ),
    )


def _make_response(persona_id: str, question_id: str = "q1") -> SurveyResponse:
    return SurveyResponse(
        persona_id=persona_id,
        question_id=question_id,
        selected_choice="A",
    )


# ──────────────────────────────────────────────
# IPF / Raking convergence tests
# ──────────────────────────────────────────────


class TestRakingConvergence:
    @pytest.mark.asyncio
    async def test_equal_distribution_preserves_unit_weights(self):
        """When sample already matches target, weights should stay ~1.0."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.TWENTIES, Gender.FEMALE),
            _make_persona("p3", AgeGroup.THIRTIES, Gender.MALE),
            _make_persona("p4", AgeGroup.THIRTIES, Gender.FEMALE),
        ]
        responses = [_make_response(p.id) for p in personas]
        config = PersonaConfig(
            panel_count=4,
            age_groups=[AgeGroup.TWENTIES, AgeGroup.THIRTIES],
            gender_ratio={"male": 0.5, "female": 0.5},
        )

        calibrator = Calibrator()
        result = await calibrator.calibrate(personas, responses, config)

        # All weights should be close to 1.0
        for p in personas:
            assert pytest.approx(p.weight, abs=0.1) == 1.0

    @pytest.mark.asyncio
    async def test_imbalanced_gender_gets_corrected(self):
        """When males are overrepresented, male weights should decrease."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p3", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p4", AgeGroup.TWENTIES, Gender.FEMALE),
        ]
        responses = [_make_response(p.id) for p in personas]
        config = PersonaConfig(
            panel_count=4,
            age_groups=[AgeGroup.TWENTIES],
            gender_ratio={"male": 0.5, "female": 0.5},
        )

        calibrator = Calibrator()
        await calibrator.calibrate(personas, responses, config)

        male_weights = [p.weight for p in personas if p.attributes.gender == Gender.MALE]
        female_weights = [p.weight for p in personas if p.attributes.gender == Gender.FEMALE]

        # Female weight should be higher than male weights to compensate
        assert all(fw > mw for fw in female_weights for mw in male_weights)

    @pytest.mark.asyncio
    async def test_weights_propagate_to_responses(self):
        """Response weights should match their persona's weight."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.TWENTIES, Gender.FEMALE),
        ]
        responses = [_make_response(p.id) for p in personas]
        config = PersonaConfig(
            panel_count=2,
            age_groups=[AgeGroup.TWENTIES],
            gender_ratio={"male": 0.5, "female": 0.5},
        )

        calibrator = Calibrator()
        result = await calibrator.calibrate(personas, responses, config)

        for resp in result:
            persona = next(p for p in personas if p.id == resp.persona_id)
            assert resp.weight == persona.weight


# ──────────────────────────────────────────────
# Weight normalization tests
# ──────────────────────────────────────────────


class TestWeightNormalization:
    def test_weights_sum_to_sample_size(self):
        """After raking, weights should sum to len(personas)."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.THIRTIES, Gender.FEMALE),
            _make_persona("p3", AgeGroup.TWENTIES, Gender.FEMALE),
            _make_persona("p4", AgeGroup.THIRTIES, Gender.MALE),
            _make_persona("p5", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p6", AgeGroup.THIRTIES, Gender.FEMALE),
        ]
        n = len(personas)
        initial = np.ones(n)
        target = {
            "age": {"20대": 0.5, "30대": 0.5},
            "gender": {"male": 0.5, "female": 0.5},
        }

        calibrator = Calibrator()
        weights = calibrator._run_raking(personas, initial, target)

        assert pytest.approx(weights.sum(), abs=1e-6) == n

    def test_convergence_within_max_iterations(self):
        """Raking should converge before hitting max_iterations for a simple case."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.TWENTIES, Gender.FEMALE),
        ]
        initial = np.ones(2)
        target = {
            "age": {"20대": 1.0},
            "gender": {"male": 0.5, "female": 0.5},
        }

        calibrator = Calibrator(max_iterations=10)
        weights = calibrator._run_raking(personas, initial, target)

        # Should converge to equal weights
        assert pytest.approx(weights[0], abs=1e-4) == weights[1]


# ──────────────────────────────────────────────
# Distribution fidelity tests
# ──────────────────────────────────────────────


class TestDistributionFidelity:
    def test_perfect_match_returns_one(self):
        """Fidelity should be 1.0 when sample perfectly matches target."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.TWENTIES, Gender.FEMALE),
        ]
        target = {
            "age": {"20대": 1.0},
            "gender": {"male": 0.5, "female": 0.5},
        }

        calibrator = Calibrator()
        fidelity = calibrator.calculate_distribution_fidelity(personas, target)
        assert pytest.approx(fidelity, abs=1e-6) == 1.0

    def test_total_mismatch_returns_low(self):
        """Fidelity should be low when sample completely mismatches target."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.TWENTIES, Gender.MALE),
        ]
        # Target expects all 30s females
        target = {
            "age": {"30대": 1.0},
            "gender": {"female": 1.0},
        }

        calibrator = Calibrator()
        fidelity = calibrator.calculate_distribution_fidelity(personas, target)
        assert fidelity < 0.5

    def test_partial_match(self):
        """Fidelity between 0 and 1 for partial match."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
            _make_persona("p2", AgeGroup.THIRTIES, Gender.FEMALE),
        ]
        target = {
            "age": {"20대": 0.5, "30대": 0.5},
            "gender": {"male": 0.5, "female": 0.5},
        }

        calibrator = Calibrator()
        fidelity = calibrator.calculate_distribution_fidelity(personas, target)
        assert pytest.approx(fidelity, abs=1e-6) == 1.0

    def test_fidelity_value_range(self):
        """Fidelity should always be between 0 and 1."""
        personas = [
            _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE),
        ]
        target = {
            "age": {"20대": 0.3, "30대": 0.7},
            "gender": {"male": 0.2, "female": 0.8},
        }

        calibrator = Calibrator()
        fidelity = calibrator.calculate_distribution_fidelity(personas, target)
        assert 0.0 <= fidelity <= 1.0


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_personas_returns_responses_unchanged(self):
        """calibrate() with empty personas should return responses as-is."""
        calibrator = Calibrator()
        responses = [_make_response("p1")]
        config = PersonaConfig(panel_count=1)

        result = await calibrator.calibrate([], responses, config)
        assert result == responses

    @pytest.mark.asyncio
    async def test_empty_responses_returns_empty(self):
        calibrator = Calibrator()
        config = PersonaConfig(panel_count=1)
        result = await calibrator.calibrate([], [], config)
        assert result == []

    @pytest.mark.asyncio
    async def test_single_persona(self):
        """Single persona should get weight = 1.0 (normalized to N=1)."""
        persona = _make_persona("p1", AgeGroup.TWENTIES, Gender.MALE)
        responses = [_make_response("p1")]
        config = PersonaConfig(
            panel_count=1,
            age_groups=[AgeGroup.TWENTIES],
            gender_ratio={"male": 1.0},
        )

        calibrator = Calibrator()
        await calibrator.calibrate([persona], responses, config)
        assert pytest.approx(persona.weight, abs=0.1) == 1.0

    def test_fidelity_empty_personas_returns_zero(self):
        calibrator = Calibrator()
        fidelity = calibrator.calculate_distribution_fidelity([], {"age": {"20대": 1.0}})
        assert fidelity == 0.0

    def test_build_target_marginals_with_age_groups(self):
        config = PersonaConfig(
            panel_count=10,
            age_groups=[AgeGroup.TWENTIES, AgeGroup.THIRTIES],
            gender_ratio={"male": 0.6, "female": 0.4},
        )
        calibrator = Calibrator()
        marginals = calibrator._build_target_marginals(config)

        assert "age" in marginals
        assert "gender" in marginals
        assert pytest.approx(sum(marginals["age"].values()), abs=1e-9) == 1.0
        assert marginals["gender"]["male"] == 0.6

    def test_build_target_marginals_without_age_groups(self):
        """When no age_groups specified, uses Korean census defaults."""
        config = PersonaConfig(panel_count=10)
        calibrator = Calibrator()
        marginals = calibrator._build_target_marginals(config)

        assert "10대" in marginals["age"]
        assert pytest.approx(sum(marginals["age"].values()), abs=1e-2) == 1.0
