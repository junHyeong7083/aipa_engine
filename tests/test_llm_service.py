"""Tests for LLMService (mock mode and retry logic)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import time

from aipa_engine.models.persona import (
    Persona,
    PersonaAttributes,
    AgeGroup,
    Gender,
)
from aipa_engine.models.survey import SurveyQuestion, QuestionType
from aipa_engine.services.llm_service import LLMService


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def persona():
    return Persona(
        id="p-test",
        name="테스트인",
        attributes=PersonaAttributes(
            age_group=AgeGroup.TWENTIES,
            gender=Gender.MALE,
            occupation="개발자",
            traits=["디지털 친숙", "실용적"],
        ),
    )


@pytest.fixture
def question():
    return SurveyQuestion(
        id="q-test",
        text="테스트 질문입니다",
        question_type=QuestionType.SINGLE_CHOICE,
        choices=["A", "B", "C"],
    )


@pytest.fixture
def likert_question():
    return SurveyQuestion(
        id="q-likert",
        text="만족도를 평가해주세요",
        question_type=QuestionType.LIKERT_SCALE,
        scale_min=1,
        scale_max=5,
        scale_labels={1: "불만족", 5: "만족"},
    )


@pytest.fixture
def open_question():
    return SurveyQuestion(
        id="q-open",
        text="자유롭게 의견을 작성해주세요",
        question_type=QuestionType.OPEN_ENDED,
    )


# ──────────────────────────────────────────────
# Mock mode tests (no API key)
# ──────────────────────────────────────────────


class TestMockMode:
    """When no API key is provided, LLMService should return mock data."""

    @patch("aipa_engine.services.llm_service.get_settings")
    def test_initializes_without_client(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="", anthropic_model="test-model"
        )
        service = LLMService(api_key="", model="test-model")
        assert service.client is None

    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_backstory_returns_mock(self, mock_settings, persona):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="", anthropic_model="test-model"
        )
        service = LLMService(api_key="", model="test-model")

        result = await service.generate_backstory(persona)

        assert persona.name in result
        assert persona.attributes.age_group.value in result
        assert persona.attributes.occupation in result

    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_explanation_returns_mock(self, mock_settings, persona, question):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="", anthropic_model="test-model"
        )
        service = LLMService(api_key="", model="test-model")

        result = await service.generate_response_explanation(
            persona, question, "A"
        )
        assert "A" in result

    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_likert_explanation_returns_mock(
        self, mock_settings, persona, likert_question
    ):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="", anthropic_model="test-model"
        )
        service = LLMService(api_key="", model="test-model")

        result = await service.generate_likert_explanation(
            persona, likert_question, 4
        )
        assert "4" in result

    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_open_response_returns_mock(
        self, mock_settings, persona, open_question
    ):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="", anthropic_model="test-model"
        )
        service = LLMService(api_key="", model="test-model")

        result = await service.generate_open_response(persona, open_question)
        assert persona.name in result


# ──────────────────────────────────────────────
# Retry logic tests
# ──────────────────────────────────────────────


class TestRetryLogic:
    """Test _call_api retry behavior with mocked anthropic client."""

    @patch("aipa_engine.services.llm_service.get_settings")
    def test_successful_call(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")

        # Mock the anthropic client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="  Hello World  ")]
        service.client = MagicMock()
        service.client.messages.create.return_value = mock_message

        result = service._call_api("test prompt", max_tokens=100)
        assert result == "Hello World"  # stripped

    @patch("aipa_engine.services.llm_service.get_settings")
    @patch("aipa_engine.services.llm_service.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep, mock_settings):
        import anthropic

        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")

        # First call rate-limited, second succeeds
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Success")]

        rate_limit_error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        service.client = MagicMock()
        service.client.messages.create.side_effect = [
            rate_limit_error,
            mock_message,
        ]

        result = service._call_api("test", max_tokens=100, max_retries=2)
        assert result == "Success"
        assert mock_sleep.call_count == 1

    @patch("aipa_engine.services.llm_service.get_settings")
    @patch("aipa_engine.services.llm_service.time.sleep")
    def test_retries_on_api_error(self, mock_sleep, mock_settings):
        import anthropic

        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Recovered")]

        api_error = anthropic.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )

        service.client = MagicMock()
        service.client.messages.create.side_effect = [
            api_error,
            mock_message,
        ]

        result = service._call_api("test", max_tokens=100, max_retries=2)
        assert result == "Recovered"

    @patch("aipa_engine.services.llm_service.get_settings")
    @patch("aipa_engine.services.llm_service.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, mock_settings):
        import anthropic

        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")

        rate_limit_error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        service.client = MagicMock()
        service.client.messages.create.side_effect = rate_limit_error

        with pytest.raises(anthropic.RateLimitError):
            service._call_api("test", max_tokens=100, max_retries=2)

        # 1 initial + 2 retries = 3 total calls
        assert service.client.messages.create.call_count == 3

    @patch("aipa_engine.services.llm_service.get_settings")
    def test_unexpected_error_raises_immediately(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")

        service.client = MagicMock()
        service.client.messages.create.side_effect = ValueError("unexpected")

        with pytest.raises(ValueError, match="unexpected"):
            service._call_api("test", max_tokens=100, max_retries=2)

        # Should not retry for unexpected errors
        assert service.client.messages.create.call_count == 1


# ──────────────────────────────────────────────
# Backstory format tests
# ──────────────────────────────────────────────


class TestBackstoryFormat:
    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_mock_backstory_contains_persona_info(self, mock_settings, persona):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="", anthropic_model="test-model"
        )
        service = LLMService(api_key="", model="test-model")

        backstory = await service.generate_backstory(persona)

        assert persona.name in backstory
        assert persona.attributes.occupation in backstory
        assert persona.attributes.age_group.value in backstory

    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_backstory_falls_back_on_api_error(self, mock_settings, persona):
        """If API call fails, backstory should fall back to mock."""
        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")
        service.client = MagicMock()
        service.client.messages.create.side_effect = Exception("API down")

        backstory = await service.generate_backstory(persona)

        # Should still return a mock backstory, not raise
        assert persona.name in backstory


# ──────────────────────────────────────────────
# Explanation generation tests
# ──────────────────────────────────────────────


class TestExplanationGeneration:
    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_explanation_falls_back_on_error(
        self, mock_settings, persona, question
    ):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")
        service.client = MagicMock()
        service.client.messages.create.side_effect = Exception("API down")

        explanation = await service.generate_response_explanation(
            persona, question, "B"
        )
        assert "B" in explanation

    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_likert_explanation_falls_back_on_error(
        self, mock_settings, persona, likert_question
    ):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")
        service.client = MagicMock()
        service.client.messages.create.side_effect = Exception("API down")

        result = await service.generate_likert_explanation(
            persona, likert_question, 3
        )
        assert "3" in result

    @patch("aipa_engine.services.llm_service.get_settings")
    @pytest.mark.asyncio
    async def test_open_response_falls_back_on_error(
        self, mock_settings, persona, open_question
    ):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="fake-key", anthropic_model="test-model"
        )
        service = LLMService(api_key="fake-key", model="test-model")
        service.client = MagicMock()
        service.client.messages.create.side_effect = Exception("API down")

        result = await service.generate_open_response(persona, open_question)
        assert persona.name in result
