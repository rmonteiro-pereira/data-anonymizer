from importlib import resources

import pandas as pd

from oblitus import Anonymizer
from oblitus.detection import scan_column
from oblitus.detection.person_names import looks_like_known_names
from oblitus.detection.place_names import is_municipality_name
from oblitus.detection.reference_lists import (
    first_names,
    municipalities,
    normalize_word,
    surnames,
)
from oblitus.taxonomy import DataCategory

MAX_PACKAGE_DATA_BYTES = 500_000


def test_the_bundled_lists_are_small_and_hold_the_expected_entries():
    data = resources.files("oblitus.detection").joinpath("data")
    total_bytes = sum(item.stat().st_size for item in data.iterdir() if item.is_file())  # type: ignore[attr-defined]

    assert total_bytes < MAX_PACKAGE_DATA_BYTES
    assert len(municipalities()) > 5000
    assert {"maria", "jose", "ana", "joao"} <= first_names()
    assert {"silva", "santos", "oliveira"} <= surnames()


def test_words_are_compared_without_case_or_accents():
    assert normalize_word("  São   PAULO ") == "sao paulo"
    assert is_municipality_name("SÃO PAULO")
    assert is_municipality_name("sao paulo")
    assert not is_municipality_name("Atlantis")


def test_a_column_of_brazilian_cities_is_a_city_without_a_name_hint():
    cities = pd.Series(["Recife", "Curitiba", "Florianópolis", "Belo Horizonte", "Natal"] * 10)

    finding = scan_column("col_3", cities)

    assert finding.category is DataCategory.CITY
    assert "Brazilian municipality" in finding.evidence


def test_a_city_column_with_foreign_values_is_still_a_city_by_name():
    cities = pd.Series(["Lisbon", "Paris", "Boston", "Osaka"] * 10)

    finding = scan_column("city", cities)

    assert finding.category is DataCategory.CITY
    assert finding.confidence < 0.6


def test_a_column_of_people_is_a_person_name_without_a_name_hint():
    people = pd.Series(["Maria Aparecida Souza", "José Carlos Lima", "Ana Paula Ferreira"] * 12)

    finding = scan_column("col_7", people)

    assert finding.category is DataCategory.PERSON_NAME
    assert looks_like_known_names(people)


def test_product_names_are_not_taken_for_people():
    products = pd.Series(
        ["Rustic Steel Chair", "Ergonomic Wooden Table", "Sleek Granite Lamp"] * 12
    )

    finding = scan_column("col_7", products)

    assert finding.category is not DataCategory.PERSON_NAME
    assert not looks_like_known_names(products)


def test_cities_found_by_value_get_a_proposal():
    frame = pd.DataFrame({"place": ["Recife", "Curitiba", "Natal", "Salvador"] * 25})
    anonymizer = Anonymizer(seed=1)

    findings = anonymizer.scan(frame)

    assert findings[0].category is DataCategory.CITY
