"""Tests for Population Generator"""

import pytest
from aipa_engine.models.persona import PersonaConfig, AgeGroup, Gender
from aipa_engine.services.population_generator import PopulationGenerator


@pytest.fixture
def generator():
    return PopulationGenerator()


@pytest.mark.asyncio
async def test_generate_default_config(generator):
    """Test persona generation with default config"""
    config = PersonaConfig(panel_count=5)
    personas = await generator.generate(config)

    assert len(personas) == 5
    for persona in personas:
        assert persona.id
        assert persona.name
        assert persona.attributes.age_group
        assert persona.attributes.gender
        assert persona.attributes.occupation


@pytest.mark.asyncio
async def test_generate_with_age_filter(generator):
    """Test persona generation with specific age groups"""
    config = PersonaConfig(
        panel_count=10,
        age_groups=[AgeGroup.TWENTIES, AgeGroup.THIRTIES],
    )
    personas = await generator.generate(config)

    assert len(personas) == 10
    for persona in personas:
        assert persona.attributes.age_group in [AgeGroup.TWENTIES, AgeGroup.THIRTIES]


@pytest.mark.asyncio
async def test_generate_with_gender_ratio(generator):
    """Test persona generation with custom gender ratio"""
    config = PersonaConfig(
        panel_count=100,
        gender_ratio={"male": 0.7, "female": 0.3},
    )
    personas = await generator.generate(config)

    male_count = sum(1 for p in personas if p.attributes.gender == Gender.MALE)
    female_count = sum(1 for p in personas if p.attributes.gender == Gender.FEMALE)

    # Allow some variance due to random sampling
    assert 55 <= male_count <= 85
    assert 15 <= female_count <= 45


@pytest.mark.asyncio
async def test_generate_with_occupations(generator):
    """Test persona generation with specific occupations"""
    config = PersonaConfig(
        panel_count=10,
        occupations=["개발자", "디자이너", "기획자"],
    )
    personas = await generator.generate(config)

    for persona in personas:
        assert persona.attributes.occupation in ["개발자", "디자이너", "기획자"]


@pytest.mark.asyncio
async def test_generate_with_traits(generator):
    """Test persona generation with specific traits"""
    config = PersonaConfig(
        panel_count=5,
        traits=["친환경", "디지털 친숙", "가성비 중시"],
    )
    personas = await generator.generate(config)

    for persona in personas:
        assert len(persona.attributes.traits) >= 2
        for trait in persona.attributes.traits:
            assert trait in ["친환경", "디지털 친숙", "가성비 중시"]
