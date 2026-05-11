# -*- coding: utf-8 -*-
"""Survey Simulation Test with Backtracking (No LLM)"""
import asyncio
import sys
import random
import numpy as np
sys.path.insert(0, r"C:\Users\user\Git\AIPA_Engine\src")

from aipa_engine.services.population_generator import PopulationGenerator
from aipa_engine.models.persona import PersonaConfig, AgeGroup

async def run_survey_simulation():
    print("=" * 70)
    print("Survey Simulation with Backtracking")
    print("=" * 70)

    # 1. Define survey questions
    questions = [
        {
            "id": "q1",
            "text": "What is your preferred shopping method?",
            "type": "single_choice",
            "choices": ["Online shopping", "Offline store", "Both equally"]
        },
        {
            "id": "q2",
            "text": "Price is more important than brand (1-5)",
            "type": "likert",
            "scale": (1, 5)
        },
        {
            "id": "q3",
            "text": "What matters most when buying electronics?",
            "type": "single_choice",
            "choices": ["Price", "Brand reputation", "Features/Specs", "Reviews"]
        },
    ]

    # 2. Generate personas
    print("\n[1] Generating Personas...")
    generator = PopulationGenerator()
    config = PersonaConfig(panel_count=5, gender_ratio={"male": 0.5, "female": 0.5})
    personas = await generator.generate(config)

    print(f"    Generated {len(personas)} personas")
    for p in personas:
        attr = p.attributes
        print(f"    - {p.name}: {attr.age_group.value}, {attr.gender.value}, {attr.occupation}")

    # 3. Generate responses based on persona attributes
    print("\n[2] Running Survey...")
    all_results = []

    for persona in personas:
        attr = persona.attributes
        responses = []

        for q in questions:
            if q["type"] == "single_choice":
                response, prob = generate_choice_response(attr, q)
            else:  # likert
                response, prob = generate_likert_response(attr, q)

            responses.append({
                "q_id": q["id"],
                "q_text": q["text"],
                "response": response,
                "probability": prob
            })

        all_results.append({
            "persona": {
                "name": persona.name,
                "age": attr.age_group.value,
                "gender": attr.gender.value,
                "occupation": attr.occupation,
                "traits": attr.traits
            },
            "responses": responses
        })

    # 4. Display results with backtracking
    print("\n[3] Results with Backtracking Analysis")
    print("=" * 70)

    for result in all_results:
        p = result["persona"]
        print(f"\n--- {p['name']} ---")
        print(f"Profile: {p['age']}, {p['gender']}, {p['occupation']}")
        print(f"Traits: {', '.join(p['traits'][:3])}")

        for resp in result["responses"]:
            print(f"\n  Q: {resp['q_text']}")
            print(f"  A: {resp['response']} (confidence: {resp['probability']:.0%})")

            # Backtracking
            backtrack = generate_backtracking(p, resp)
            print(f"  >> Why: {backtrack}")

    # 5. Aggregate statistics
    print("\n" + "=" * 70)
    print("[4] Aggregate Results")
    print("=" * 70)

    for q in questions:
        print(f"\nQ: {q['text']}")
        responses_for_q = [
            r["response"]
            for result in all_results
            for r in result["responses"]
            if r["q_id"] == q["id"]
        ]

        if q["type"] == "single_choice":
            counts = {}
            for r in responses_for_q:
                counts[r] = counts.get(r, 0) + 1
            for choice, count in counts.items():
                pct = count / len(responses_for_q) * 100
                bar = "#" * int(pct / 5)
                print(f"   {choice}: {count} ({pct:.0f}%) {bar}")
        else:
            avg = sum(responses_for_q) / len(responses_for_q)
            print(f"   Average: {avg:.1f} / 5")
            dist = {i: responses_for_q.count(i) for i in range(1, 6)}
            print(f"   Distribution: {dist}")

    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)


def generate_choice_response(attr, question):
    """Generate single choice response based on persona attributes"""
    age = attr.age_group
    traits = attr.traits or []

    q_id = question["id"]
    choices = question["choices"]

    # Build probability distribution based on persona
    probs = [1.0] * len(choices)

    if q_id == "q1":  # Shopping method
        # Age effect
        if age in [AgeGroup.TEENS, AgeGroup.TWENTIES]:
            probs[0] *= 2.5  # Online
            probs[1] *= 0.5  # Offline
        elif age in [AgeGroup.FIFTIES, AgeGroup.SIXTIES_PLUS]:
            probs[0] *= 0.7  # Online
            probs[1] *= 2.0  # Offline

        # Trait effect
        if any(t in ["디지털 친숙", "트렌디"] for t in traits):
            probs[0] *= 1.5
        if any(t in ["오프라인 선호", "보수적"] for t in traits):
            probs[1] *= 1.5

    elif q_id == "q3":  # Electronics purchase
        # Age effect
        if age in [AgeGroup.TWENTIES, AgeGroup.THIRTIES]:
            probs[2] *= 1.8  # Features
            probs[3] *= 1.5  # Reviews
        elif age == AgeGroup.SIXTIES_PLUS:
            probs[0] *= 1.5  # Price
            probs[1] *= 1.3  # Brand

        # Trait effect
        if any(t in ["가성비 중시", "실용적"] for t in traits):
            probs[0] *= 2.0  # Price
        if any(t in ["브랜드 선호", "품질 중시"] for t in traits):
            probs[1] *= 2.0  # Brand

    # Normalize and sample
    total = sum(probs)
    probs = [p / total for p in probs]
    idx = np.random.choice(len(choices), p=probs)

    return choices[idx], probs[idx]


def generate_likert_response(attr, question):
    """Generate Likert scale response based on persona attributes"""
    age = attr.age_group
    traits = attr.traits or []

    # Base: normal distribution around 3
    mean = 3.0
    std = 1.0

    # Q2: Price vs Brand
    if question["id"] == "q2":
        # Age effect (younger = more price sensitive)
        if age in [AgeGroup.TEENS, AgeGroup.TWENTIES]:
            mean += 0.8
        elif age == AgeGroup.SIXTIES_PLUS:
            mean += 0.5
        elif age in [AgeGroup.THIRTIES, AgeGroup.FORTIES]:
            mean -= 0.3

        # Trait effect
        if any(t in ["가성비 중시", "실용적"] for t in traits):
            mean += 1.0
        if any(t in ["브랜드 선호", "품질 중시"] for t in traits):
            mean -= 1.0

    # Sample and clamp
    value = int(round(np.random.normal(mean, std)))
    value = max(1, min(5, value))

    # Calculate probability (higher if closer to mean)
    prob = 1.0 - abs(value - mean) / 4

    return value, prob


def generate_backtracking(persona, response):
    """Generate explanation for why persona gave this response"""
    age = persona["age"]
    traits = persona["traits"]
    occupation = persona["occupation"]
    resp = response["response"]
    q_id = response["q_id"]

    reasons = []

    if q_id == "q1":  # Shopping method
        if "Online" in str(resp):
            if age in ["10대", "20대"]:
                reasons.append(f"Young ({age}) prefers digital channels")
            if "디지털 친숙" in traits or "트렌디" in traits:
                reasons.append("Tech-savvy/trendy personality")
        elif "Offline" in str(resp):
            if age in ["50대", "60대+"]:
                reasons.append(f"Older ({age}) values in-person experience")
            if "보수적" in traits or "오프라인 선호" in traits:
                reasons.append("Prefers traditional shopping")
        else:
            reasons.append("Balanced approach to shopping")

    elif q_id == "q2":  # Price importance
        if resp >= 4:
            if "가성비 중시" in traits:
                reasons.append("Value-conscious trait")
            if age in ["10대", "20대", "60대+"]:
                reasons.append(f"Age group ({age}) typically budget-conscious")
        elif resp <= 2:
            if "브랜드 선호" in traits or "품질 중시" in traits:
                reasons.append("Quality/brand focused")
        else:
            reasons.append("Balanced price-quality consideration")

    elif q_id == "q3":  # Electronics factor
        if resp == "Price":
            reasons.append("Cost is primary concern" if "가성비" in str(traits) else "Budget-driven decision")
        elif resp == "Brand reputation":
            reasons.append("Trust established brands")
        elif resp == "Features/Specs":
            reasons.append("Research-oriented buyer")
        elif resp == "Reviews":
            reasons.append("Values peer opinions")

    if not reasons:
        reasons.append(f"General {age} {occupation} behavior pattern")

    return " | ".join(reasons)


if __name__ == "__main__":
    asyncio.run(run_survey_simulation())
