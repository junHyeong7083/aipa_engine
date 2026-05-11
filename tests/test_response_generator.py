"""Tests for ResponseGenerator service."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from aipa_engine.models.persona import (
    Persona,
    PersonaAttributes,
    AgeGroup,
    Gender,
)
from aipa_engine.models.survey import SurveyQuestion, SurveyResponse, QuestionType
from aipa_engine.services.response_generator import ResponseGenerator


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_generator(mock_llm) -> ResponseGenerator:
    """Create a ResponseGenerator wired to a mock LLM service."""
    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.llm_service = mock_llm
    gen._setup_priors()
    return gen


# ──────────────────────────────────────────────
# Single-choice response tests
# ──────────────────────────────────────────────


class TestSingleChoiceResponse:
    @pytest.mark.asyncio
    async def test_returns_survey_response(
        self, sample_persona, single_choice_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        response = await gen.generate_response(sample_persona, single_choice_question)

        assert isinstance(response, SurveyResponse)
        assert response.persona_id == sample_persona.id
        assert response.question_id == single_choice_question.id
        assert response.selected_choice in single_choice_question.choices

    @pytest.mark.asyncio
    async def test_probability_is_valid(
        self, sample_persona, single_choice_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        response = await gen.generate_response(sample_persona, single_choice_question)

        assert 0.0 < response.probability <= 1.0

    @pytest.mark.asyncio
    async def test_explanation_generated(
        self, sample_persona, single_choice_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        response = await gen.generate_response(
            sample_persona, single_choice_question, generate_explanation=True
        )
        assert response.explanation is not None
        mock_llm_service.generate_response_explanation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_explanation_when_disabled(
        self, sample_persona, single_choice_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        response = await gen.generate_response(
            sample_persona, single_choice_question, generate_explanation=False
        )
        assert response.explanation is None
        mock_llm_service.generate_response_explanation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_choices_returns_none(
        self, sample_persona, mock_llm_service
    ):
        question = SurveyQuestion(
            id="q-empty",
            text="No choices here",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=[],
        )
        gen = _make_generator(mock_llm_service)
        response = await gen.generate_response(sample_persona, question)
        assert response.selected_choice is None


# ──────────────────────────────────────────────
# Probability calculation tests
# ──────────────────────────────────────────────


class TestProbabilityCalculation:
    def test_probabilities_sum_to_one(
        self, sample_persona, single_choice_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        probs = gen._calculate_choice_probabilities(
            sample_persona, single_choice_question
        )
        assert pytest.approx(sum(probs), abs=1e-9) == 1.0

    def test_uniform_when_no_keywords(self, mock_llm_service):
        """Choices with no matching keywords should stay roughly uniform."""
        persona = Persona(
            id="p-neutral",
            name="테스트",
            attributes=PersonaAttributes(
                age_group=AgeGroup.THIRTIES,
                gender=Gender.MALE,
                occupation="기타",
                traits=[],
            ),
        )
        question = SurveyQuestion(
            id="q-neutral",
            text="선호하는 색상은?",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=["빨강", "파랑", "초록"],
        )
        gen = _make_generator(mock_llm_service)
        probs = gen._calculate_choice_probabilities(persona, question)

        # All should be equal (1/3 each) since no keywords match
        assert all(pytest.approx(p, abs=1e-9) == 1 / 3 for p in probs)

    def test_price_keyword_boosts_for_price_sensitive_persona(self, mock_llm_service):
        """A young (price-sensitive) persona should have higher probability for price choices."""
        teen = Persona(
            id="p-teen",
            name="청소년",
            attributes=PersonaAttributes(
                age_group=AgeGroup.TEENS,
                gender=Gender.MALE,
                occupation="학생",
                traits=["가격 민감"],
            ),
        )
        question = SurveyQuestion(
            id="q-price",
            text="중요한 것은?",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=["저렴한 가격", "높은 품질"],
        )
        gen = _make_generator(mock_llm_service)
        probs = gen._calculate_choice_probabilities(teen, question)

        # "저렴한 가격" should have higher probability due to price sensitivity + trait
        assert probs[0] > probs[1]

    def test_tech_keyword_boosts_for_young_persona(self, mock_llm_service):
        """A 20s persona with tech traits should prefer tech choices."""
        persona = Persona(
            id="p-tech",
            name="테크인",
            attributes=PersonaAttributes(
                age_group=AgeGroup.TWENTIES,
                gender=Gender.FEMALE,
                occupation="개발자",
                traits=["얼리어답터"],
            ),
        )
        question = SurveyQuestion(
            id="q-tech",
            text="선호하는 쇼핑 방법은?",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=["모바일 앱 주문", "매장 방문"],
        )
        gen = _make_generator(mock_llm_service)
        probs = gen._calculate_choice_probabilities(persona, question)

        # "모바일 앱 주문" should have higher probability
        assert probs[0] > probs[1]

    def test_trait_modifier_applies_correctly(self, mock_llm_service):
        """Trait '친환경 선호' should boost eco-keyword choices."""
        persona = Persona(
            id="p-eco",
            name="에코",
            attributes=PersonaAttributes(
                age_group=AgeGroup.THIRTIES,
                gender=Gender.FEMALE,
                occupation="교사",
                traits=["친환경 선호"],
            ),
        )
        question = SurveyQuestion(
            id="q-eco",
            text="구매 기준은?",
            question_type=QuestionType.SINGLE_CHOICE,
            choices=["유기농 제품", "일반 제품"],
        )
        gen = _make_generator(mock_llm_service)
        probs = gen._calculate_choice_probabilities(persona, question)

        assert probs[0] > probs[1]


# ──────────────────────────────────────────────
# Keyword matching tests
# ──────────────────────────────────────────────


class TestKeywordMatching:
    def test_price_synonyms(self, mock_llm_service):
        gen = _make_generator(mock_llm_service)
        assert gen._matches_keyword_category("저렴한 상품", "가격") is True
        assert gen._matches_keyword_category("할인 이벤트", "가격") is True
        assert gen._matches_keyword_category("고급 소재", "가격") is False

    def test_tech_synonyms(self, mock_llm_service):
        gen = _make_generator(mock_llm_service)
        assert gen._matches_keyword_category("모바일 앱", "기술") is True
        assert gen._matches_keyword_category("ai 기반", "기술") is True
        assert gen._matches_keyword_category("수제 빵", "기술") is False

    def test_unknown_category_uses_literal(self, mock_llm_service):
        gen = _make_generator(mock_llm_service)
        assert gen._matches_keyword_category("특별한 단어", "특별한") is True
        assert gen._matches_keyword_category("다른 단어", "특별한") is False


# ──────────────────────────────────────────────
# Likert-scale response tests
# ──────────────────────────────────────────────


class TestLikertResponse:
    @pytest.mark.asyncio
    async def test_returns_valid_scale_value(
        self, sample_persona, likert_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        response = await gen.generate_response(sample_persona, likert_question)

        assert response.scale_value is not None
        assert likert_question.scale_min <= response.scale_value <= likert_question.scale_max

    @pytest.mark.asyncio
    async def test_likert_explanation_called(
        self, sample_persona, likert_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        await gen.generate_response(
            sample_persona, likert_question, generate_explanation=True
        )
        mock_llm_service.generate_likert_explanation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_likert_values_within_range_over_many_runs(
        self, sample_persona, likert_question, mock_llm_service
    ):
        """Generate many responses and verify all are within scale bounds."""
        gen = _make_generator(mock_llm_service)
        for _ in range(50):
            resp = await gen.generate_response(
                sample_persona, likert_question, generate_explanation=False
            )
            assert likert_question.scale_min <= resp.scale_value <= likert_question.scale_max


# ──────────────────────────────────────────────
# Open-ended response tests
# ──────────────────────────────────────────────


class TestOpenEndedResponse:
    @pytest.mark.asyncio
    async def test_returns_open_response(
        self, sample_persona, open_ended_question, mock_llm_service
    ):
        gen = _make_generator(mock_llm_service)
        response = await gen.generate_response(sample_persona, open_ended_question)

        assert response.open_response is not None
        mock_llm_service.generate_open_response.assert_awaited_once()
