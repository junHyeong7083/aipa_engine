"""Persona Generation Test"""
import asyncio
import sys
sys.path.insert(0, r"C:\Users\user\Git\AIPA_Engine\src")

from aipa_engine.services.population_generator import PopulationGenerator
from aipa_engine.models.persona import PersonaConfig

async def test_persona_generation():
    print("=" * 60)
    print("Persona Generation Test")
    print("=" * 60)

    generator = PopulationGenerator()

    # 10명의 페르소나 생성
    config = PersonaConfig(
        panel_count=10,
        gender_ratio={"male": 0.5, "female": 0.5},
    )

    print("\nGenerating 10 personas...")
    personas = await generator.generate(config)

    print(f"\nGenerated {len(personas)} personas:\n")
    print("-" * 60)

    for i, persona in enumerate(personas, 1):
        attr = persona.attributes
        print(f"{i}. {persona.name}")
        print(f"   Age: {attr.age_group.value if attr.age_group else 'N/A'}")
        print(f"   Gender: {attr.gender.value if attr.gender else 'N/A'}")
        print(f"   Occupation: {attr.occupation or 'N/A'}")
        print(f"   Traits: {', '.join(attr.traits) if attr.traits else 'N/A'}")
        print()

    # 통계 요약
    print("-" * 60)
    print("Summary Statistics:")

    # 성별 분포
    male_count = sum(1 for p in personas if p.attributes.gender.value == "male")
    female_count = len(personas) - male_count
    print(f"  Gender: Male {male_count}, Female {female_count}")

    # 연령대 분포
    age_counts = {}
    for p in personas:
        age = p.attributes.age_group.value if p.attributes.age_group else "unknown"
        age_counts[age] = age_counts.get(age, 0) + 1
    print(f"  Age groups: {age_counts}")

    # 직업 분포
    occ_counts = {}
    for p in personas:
        occ = p.attributes.occupation or "unknown"
        occ_counts[occ] = occ_counts.get(occ, 0) + 1
    print(f"  Occupations: {occ_counts}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_persona_generation())
