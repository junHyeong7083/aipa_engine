"""
Shared pytest fixtures for AIPA Engine tests.

Provides reusable sample data and mock services used across all test modules.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aipa_engine.models.persona import (
    Persona,
    PersonaAttributes,
    PersonaConfig,
    AgeGroup,
    Gender,
)
from aipa_engine.models.survey import SurveyQuestion, SurveyResponse, QuestionType


# ──────────────────────────────────────────────
# Persona fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_persona() -> Persona:
    """A single 20s male office worker persona."""
    return Persona(
        id="persona-001",
        name="김민준",
        attributes=PersonaAttributes(
            age_group=AgeGroup.TWENTIES,
            gender=Gender.MALE,
            occupation="회사원",
            traits=["가성비 중시", "디지털 친숙"],
            interests=["게임", "여행"],
        ),
    )


@pytest.fixture
def sample_persona_female() -> Persona:
    """A single 40s female professional persona."""
    return Persona(
        id="persona-002",
        name="이서연",
        attributes=PersonaAttributes(
            age_group=AgeGroup.FORTIES,
            gender=Gender.FEMALE,
            occupation="전문직",
            traits=["품질 중시", "건강 관심"],
            interests=["요리", "독서"],
        ),
    )


@pytest.fixture
def sample_personas(sample_persona, sample_persona_female) -> list[Persona]:
    """A small list of diverse personas for calibration tests."""
    extra = [
        Persona(
            id="persona-003",
            name="박지훈",
            attributes=PersonaAttributes(
                age_group=AgeGroup.THIRTIES,
                gender=Gender.MALE,
                occupation="개발자",
                traits=["얼리어답터", "실용적"],
            ),
        ),
        Persona(
            id="persona-004",
            name="최수진",
            attributes=PersonaAttributes(
                age_group=AgeGroup.TWENTIES,
                gender=Gender.FEMALE,
                occupation="디자이너",
                traits=["트렌디", "친환경 선호"],
            ),
        ),
    ]
    return [sample_persona, sample_persona_female] + extra


# ──────────────────────────────────────────────
# Question fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def single_choice_question() -> SurveyQuestion:
    """A single-choice question with price/tech keywords."""
    return SurveyQuestion(
        id="q-sc-001",
        text="온라인 쇼핑에서 가장 중요하게 생각하는 것은?",
        question_type=QuestionType.SINGLE_CHOICE,
        choices=["저렴한 가격", "빠른 배송", "앱 편의성", "브랜드 인지도"],
    )


@pytest.fixture
def likert_question() -> SurveyQuestion:
    """A Likert-scale question (1-5)."""
    return SurveyQuestion(
        id="q-lk-001",
        text="친환경 제품에 더 높은 가격을 지불할 의향이 있습니까?",
        question_type=QuestionType.LIKERT_SCALE,
        scale_min=1,
        scale_max=5,
        scale_labels={1: "전혀 아니다", 3: "보통", 5: "매우 그렇다"},
    )


@pytest.fixture
def open_ended_question() -> SurveyQuestion:
    """An open-ended question."""
    return SurveyQuestion(
        id="q-oe-001",
        text="최근 구매한 제품에 대해 자유롭게 의견을 작성해주세요.",
        question_type=QuestionType.OPEN_ENDED,
    )


# ──────────────────────────────────────────────
# Config fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_config() -> PersonaConfig:
    """A basic PersonaConfig for 10 personas across 20s/30s."""
    return PersonaConfig(
        panel_count=10,
        age_groups=[AgeGroup.TWENTIES, AgeGroup.THIRTIES],
        gender_ratio={"male": 0.5, "female": 0.5},
        occupations=["회사원", "개발자"],
        traits=["가성비 중시", "디지털 친숙"],
    )


# ──────────────────────────────────────────────
# Mock LLM service fixture
# ──────────────────────────────────────────────


@pytest.fixture
def mock_llm_service():
    """
    A fully mocked LLMService that returns deterministic strings
    without requiring any API key or network access.
    """
    service = MagicMock()
    service.client = None  # mock mode

    service.generate_backstory = AsyncMock(
        return_value="Mock 배경 스토리입니다."
    )
    service.generate_response_explanation = AsyncMock(
        return_value="Mock 설명입니다."
    )
    service.generate_likert_explanation = AsyncMock(
        return_value="3점을 선택했습니다."
    )
    service.generate_open_response = AsyncMock(
        return_value="Mock 주관식 응답입니다."
    )
    return service
