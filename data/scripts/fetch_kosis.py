"""
KOSIS Data Fetcher Script

Fetches statistical data from KOSIS Open API and saves to processed folder.

Usage:
    python data/scripts/fetch_kosis.py --api-key YOUR_API_KEY

Or set KOSIS_API_KEY environment variable.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx


KOSIS_BASE_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

# Important KOSIS table IDs for persona generation
TABLES = {
    "population_age_5year": {
        "tblId": "DT_1B040M5",
        "orgId": "101",
        "description": "주민등록인구 5세 단위 연령별",
    },
    "population_gender_region": {
        "tblId": "DT_1B040M1",
        "orgId": "101",
        "description": "주민등록인구 시도별 성별",
    },
    "employed_by_occupation": {
        "tblId": "DT_1DA7012S",
        "orgId": "101",
        "description": "직업별 취업자",
    },
    "population_education": {
        "tblId": "DT_1PM1502",
        "orgId": "101",
        "description": "학력별 인구",
    },
}


async def fetch_table(api_key: str, table_info: dict, year: str = "2023") -> dict:
    """Fetch a single KOSIS table"""
    params = {
        "method": "getList",
        "apiKey": api_key,
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "Y",
        "startPrdDe": year,
        "endPrdDe": year,
        "orgId": table_info["orgId"],
        "tblId": table_info["tblId"],
    }

    async with httpx.AsyncClient() as client:
        print(f"Fetching {table_info['description']}...")
        response = await client.get(KOSIS_BASE_URL, params=params, timeout=30.0)
        response.raise_for_status()
        return response.json()


async def main(api_key: str, output_dir: Path, year: str = "2023"):
    """Fetch all tables and save to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table_info in TABLES.items():
        try:
            data = await fetch_table(api_key, table_info, year)

            # Save raw data
            output_file = output_dir / f"{name}_{year}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": f"KOSIS {table_info['tblId']}",
                        "description": table_info["description"],
                        "fetched_at": datetime.now().isoformat(),
                        "year": year,
                        "data": data,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  Saved to {output_file}")

        except Exception as e:
            print(f"  Error fetching {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch KOSIS statistical data")
    parser.add_argument("--api-key", help="KOSIS API key")
    parser.add_argument("--year", default="2023", help="Year to fetch (default: 2023)")
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory (default: data/raw)",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("KOSIS_API_KEY")
    if not api_key:
        print("Error: KOSIS API key required. Use --api-key or set KOSIS_API_KEY env var")
        print("\nGet your API key from: https://kosis.kr/openapi/")
        exit(1)

    output_dir = Path(args.output)
    asyncio.run(main(api_key, output_dir, args.year))
    print("\nDone!")
