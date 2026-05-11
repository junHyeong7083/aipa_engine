"""Tests for SimulationService orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aipa_engine.models.persona import (
    Persona,
    PersonaAttributes,
    PersonaConfig,
    AgeGroup,
    Gender,
)
from aipa_engine.models.survey import SurveyQuestion, SurveyResponse, QuestionType
from aipa_engine.services.simulation_service import SimulationService


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _build_service(
    mock_pop_gen=None, mock_resp_gen=None, mock_calibrator=None, mock_llm=None
) -> SimulationService:
    """Build a SimulationService with fully mocked sub-services."""
    svc = SimulationService.__new__(SimulationService)
    svc.llm_service = mock_llm or MagicMock()
    svc.population_generator = mock_pop_gen or MagicMock()
    svc.response_generator = mock_resp_gen or MagicMock()
    svc.calibrator = mock_calibrator or MagicMock()
    return svc


def _sample_personas() -> list[Persona]:
    return [
        Persona(
            id="p1",
            name="김민준",
            attributes=PersonaAttributes(
                age_group=AgeGroup.TWENTIES,
                gender=Gender.MALE,
                occupation="회사원",
            ),
        ),
        Persona(
            id="p2",
            name="이서연",
            attributes=PersonaAttributes(
                age_group=AgeGroup.THIRTIES,
                gender=Gender.FEMALE,
                occupation="디자이너",
            ),
        ),
    ]


def _sample_questions() -> list[SurveyQuestion]:
    return [
        SurveyQuestion(
            id="q1",
            text="좋아하는 색상은?",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=["빨강", "파랑"],
        ),
    ]


# ──────────────────────────────────────────────
# generate_personas tests
# ──────────────────────────────────────────────


class TestGeneratePersonas:
    @pytest.mark.asyncio
    async def test_delegates_to_population_generator(self):
        personas = _sample_personas()
        mock_pop = MagicMock()
        mock_pop.generate = AsyncMock(return_value=personas)

        svc = _build_service(mock_pop_gen=mock_pop)
        config = PersonaConfig(panel_count=2)

        result = await svc.generate_personas(config)

        mock_pop.generate.assert_awaited_once_with(config)
        assert result == personas

    @pytest.mark.asyncio
    async def test_backstories_generated_when_requested(self):
        personas = _sample_personas()
        mock_pop = MagicMock()
        mock_pop.generate = AsyncMock(return_value=personas)
        mock_llm = MagicMock()
        mock_llm.generate_backstory = AsyncMock(return_value="배경 스토리")

        svc = _build_service(mock_pop_gen=mock_pop, mock_llm=mock_llm)
        config = PersonaConfig(panel_count=2)

        result = await svc.generate_personas(config, generate_backstories=True)

        assert mock_llm.generate_backstory.await_count == len(personas)
        for p in result:
            assert p.backstory == "배경 스토리"

    @pytest.mark.asyncio
    async def test_no_backstories_by_default(self):
        personas = _sample_personas()
        mock_pop = MagicMock()
        mock_pop.generate = AsyncMock(return_value=personas)
        mock_llm = MagicMock()

        svc = _build_service(mock_pop_gen=mock_pop, mock_llm=mock_llm)
        config = PersonaConfig(panel_count=2)

        await svc.generate_personas(config)

        mock_llm.generate_backstory.assert_not_called()


# ──────────────────────────────────────────────
# run_survey tests
# ──────────────────────────────────────────────


class TestRunSurvey:
    @pytest.mark.asyncio
    async def test_generates_response_for_each_combination(self):
        personas = _sample_personas()
        questions = _sample_questions()

        mock_resp = MagicMock()
        mock_resp.generate_response = AsyncMock(
            return_value=SurveyResponse(
                persona_id="p1", question_id="q1", selected_choice="빨강"
            )
        )

        svc = _build_service(mock_resp_gen=mock_resp)
        result = await svc.run_survey(personas, questions)

        # 2 personas * 1 question = 2 calls
        assert mock_resp.generate_response.await_count == 2
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_personas_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.generate_response = AsyncMock()

        svc = _build_service(mock_resp_gen=mock_resp)
        result = await svc.run_survey([], _sample_questions())

        assert result == []
        mock_resp.generate_response.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_questions_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.generate_response = AsyncMock()

        svc = _build_service(mock_resp_gen=mock_resp)
        result = await svc.run_survey(_sample_personas(), [])

        assert result == []


# ──────────────────────────────────────────────
# calibrate tests
# ──────────────────────────────────────────────


class TestCalibrate:
    @pytest.mark.asyncio
    async def test_delegates_to_calibrator(self):
        personas = _sample_personas()
        responses = [
            SurveyResponse(persona_id="p1", question_id="q1", selected_choice="A")
        ]
        config = PersonaConfig(panel_count=2)

        mock_cal = MagicMock()
        mock_cal.calibrate = AsyncMock(return_value=responses)

        svc = _build_service(mock_calibrator=mock_cal)
        result = await svc.calibrate(responses, config, personas)

        mock_cal.calibrate.assert_awaited_once_with(personas, responses, config)
        assert result == responses

    @pytest.mark.asyncio
    async def test_no_personas_skips_calibration(self):
        responses = [
            SurveyResponse(persona_id="p1", question_id="q1", selected_choice="A")
        ]
        config = PersonaConfig(panel_count=1)

        mock_cal = MagicMock()
        svc = _build_service(mock_calibrator=mock_cal)
        result = await svc.calibrate(responses, config, personas=None)

        mock_cal.calibrate.assert_not_called()
        assert result == responses


# ──────────────────────────────────────────────
# run_full_simulation tests
# ──────────────────────────────────────────────


class TestRunFullSimulation:
    @pytest.mark.asyncio
    async def test_full_pipeline_executes_all_stages(self):
        personas = _sample_personas()
        questions = _sample_questions()
        responses = [
            SurveyResponse(persona_id="p1", question_id="q1", selected_choice="빨강"),
            SurveyResponse(persona_id="p2", question_id="q1", selected_choice="파랑"),
        ]
        calibrated = [
            SurveyResponse(persona_id="p1", question_id="q1", selected_choice="빨강", weight=1.2),
            SurveyResponse(persona_id="p2", question_id="q1", selected_choice="파랑", weight=0.8),
        ]

        mock_pop = MagicMock()
        mock_pop.generate = AsyncMock(return_value=personas)
        mock_resp = MagicMock()
        mock_resp.generate_response = AsyncMock(side_effect=responses)
        mock_cal = MagicMock()
        mock_cal.calibrate = AsyncMock(return_value=calibrated)

        svc = _build_service(
            mock_pop_gen=mock_pop,
            mock_resp_gen=mock_resp,
            mock_calibrator=mock_cal,
        )

        config = PersonaConfig(panel_count=2)
        result_personas, result_responses = await svc.run_full_simulation(
            config, questions
        )

        assert result_personas == personas
        assert result_responses == calibrated

    @pytest.mark.asyncio
    async def test_calibration_skipped_when_disabled(self):
        personas = _sample_personas()
        questions = _sample_questions()
        dummy_response = SurveyResponse(
            persona_id="p1", question_id="q1", selected_choice="빨강"
        )

        mock_pop = MagicMock()
        mock_pop.generate = AsyncMock(return_value=personas)
        mock_resp = MagicMock()
        # return_value (not side_effect) so it works for any number of calls
        mock_resp.generate_response = AsyncMock(return_value=dummy_response)
        mock_cal = MagicMock()
        mock_cal.calibrate = AsyncMock()

        svc = _build_service(
            mock_pop_gen=mock_pop,
            mock_resp_gen=mock_resp,
            mock_calibrator=mock_cal,
        )

        config = PersonaConfig(panel_count=2)
        await svc.run_full_simulation(
            config, questions, enable_calibration=False
        )

        mock_cal.calibrate.assert_not_awaited()


# ──────────────────────────────────────────────
# Error handling tests
# ──────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_population_generator_failure_propagates(self):
        mock_pop = MagicMock()
        mock_pop.generate = AsyncMock(side_effect=RuntimeError("Generation failed"))

        svc = _build_service(mock_pop_gen=mock_pop)
        config = PersonaConfig(panel_count=2)

        with pytest.raises(RuntimeError, match="Generation failed"):
            await svc.generate_personas(config)

    @pytest.mark.asyncio
    async def test_response_generator_failure_propagates(self):
        personas = _sample_personas()
        questions = _sample_questions()

        mock_resp = MagicMock()
        mock_resp.generate_response = AsyncMock(
            side_effect=RuntimeError("LLM failed")
        )

        svc = _build_service(mock_resp_gen=mock_resp)

        with pytest.raises(RuntimeError, match="LLM failed"):
            await svc.run_survey(personas, questions)

    @pytest.mark.asyncio
    async def test_calibrator_failure_propagates(self):
        personas = _sample_personas()
        responses = [
            SurveyResponse(persona_id="p1", question_id="q1", selected_choice="A")
        ]
        config = PersonaConfig(panel_count=2)

        mock_cal = MagicMock()
        mock_cal.calibrate = AsyncMock(
            side_effect=RuntimeError("Calibration diverged")
        )

        svc = _build_service(mock_calibrator=mock_cal)

        with pytest.raises(RuntimeError, match="Calibration diverged"):
            await svc.calibrate(responses, config, personas)
