"""KOSIS Client Full Integration Test"""
import asyncio
import sys
import os

sys.path.insert(0, r"C:\Users\user\Git\AIPA_Engine\src")

from aipa_engine.services.kosis_client import KOSISClient

async def test_all():
    print("=" * 60)
    print("KOSIS Client Full Integration Test")
    print("=" * 60)

    api_key = os.environ.get("KOSIS_API_KEY", "")
    if not api_key:
        print("[ERROR] KOSIS_API_KEY not set. Export it or add to .env")
        return

    client = KOSISClient(
        api_key=api_key,
        cache_dir=r"C:\Users\user\Git\AIPA_Engine\data\processed"
    )

    # 1. 연령별 분포
    print("\n1. Age Distribution...")
    try:
        age_dist = await client.get_age_distribution(force_refresh=True)
        print("   [OK]")
        for age, ratio in age_dist.items():
            print(f"      {age}: {ratio*100:.1f}%")
    except Exception as e:
        print(f"   [FAIL]: {e}")

    # 2. 성별 분포
    print("\n2. Gender Distribution...")
    try:
        gender_dist = await client.get_gender_distribution(force_refresh=True)
        print("   [OK]")
        for gender, ratio in gender_dist.items():
            print(f"      {gender}: {ratio*100:.1f}%")
    except Exception as e:
        print(f"   [FAIL]: {e}")

    # 3. 직업별 분포
    print("\n3. Occupation Distribution...")
    try:
        occ_dist = await client.get_occupation_distribution(force_refresh=True)
        print("   [OK]")
        for occ, ratio in occ_dist.items():
            print(f"      {occ}: {ratio*100:.1f}%")
    except Exception as e:
        print(f"   [FAIL]: {e}")
        import traceback
        traceback.print_exc()

    # 4. 캐시 파일 확인
    print("\n4. Cache files...")
    import json
    from pathlib import Path

    cache_dir = Path(r"C:\Users\user\Git\AIPA_Engine\data\processed")
    for cache_file in ["age_distribution.json", "gender_distribution.json", "occupation_distribution.json"]:
        file_path = cache_dir / cache_file
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"   [OK] {cache_file}")
            print(f"        source: {cached.get('source', 'N/A')}")
        else:
            print(f"   [MISSING] {cache_file}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all())
