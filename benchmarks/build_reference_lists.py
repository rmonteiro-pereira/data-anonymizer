"""Rebuild the reference lists in `src/oblitus/detection/data/` from IBGE public data.

Run it with network access: `uv run python benchmarks/build_reference_lists.py`.
- `municipalities.txt` holds the names of the 5,570 Brazilian municipalities.
- `first_names.txt` holds the most frequent first names of the 2010 Census, collected from the
  names ranking for every decade, sex and state, because the API returns 20 names per query.
- `surnames.txt` holds common Brazilian surnames. IBGE does not publish a surname ranking,
  so they come from the Faker `pt_BR` provider.
"""

import gzip
import json
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

from faker.providers.person.pt_BR import Provider as BrazilianPersonProvider

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "oblitus" / "detection" / "data"
MUNICIPALITIES_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
RANKING_URL = "https://servicodados.ibge.gov.br/api/v2/censos/nomes/ranking"
DECADES = (1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010)
STATE_CODES = (
    list(range(11, 18)) + list(range(21, 30)) + [31, 32, 33, 35] + [41, 42, 43, 50, 51, 52, 53]
)
MIN_NAME_LENGTH = 3
GZIP_MAGIC = b"\x1f\x8b"


def fetch_json(url: str) -> list:
    with urllib.request.urlopen(url, timeout=20) as response:
        body = response.read()
    if body[:2] == GZIP_MAGIC:
        body = gzip.decompress(body)
    return json.loads(body)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def ranking_names(query: str) -> list[str]:
    """Return the names of one ranking query, or none when IBGE has no data for it."""
    try:
        ranking = fetch_json(f"{RANKING_URL}?{query}")
    except OSError:
        return []
    if not ranking:
        return []
    return [item["nome"] for item in ranking[0]["res"]]


def build_municipalities() -> list[dict]:
    return fetch_json(MUNICIPALITIES_URL)


def build_first_names(municipalities: list[dict]) -> list[str]:
    """Collect the top 20 names for every state and decade, and for every municipality."""
    queries = [
        f"decada={decade}&sexo={sex}&localidade={state}"
        for decade, sex, state in product(DECADES, ("M", "F"), STATE_CODES)
    ]
    for municipality in municipalities:
        for sex in ("M", "F"):
            queries.append(f"sexo={sex}&localidade={municipality['id']}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(ranking_names, queries))
    names = {name.lower() for batch in batches for name in batch if len(name) >= MIN_NAME_LENGTH}
    return sorted(names)


def build_surnames() -> list[str]:
    names = {strip_accents(name).lower() for name in BrazilianPersonProvider.last_names}
    return sorted(name for name in names if " " not in name)


def write_lines(filename: str, lines: list[str]) -> None:
    (DATA_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{filename}: {len(lines)} entries")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    municipalities = build_municipalities()
    write_lines("municipalities.txt", sorted({item["nome"] for item in municipalities}))
    write_lines("first_names.txt", build_first_names(municipalities))
    write_lines("surnames.txt", build_surnames())


if __name__ == "__main__":
    main()
