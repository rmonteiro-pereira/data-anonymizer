"""A labelled benchmark of columns for the detector, generated the same way every time.

Each column has a known category. The columns cover clean identifiers, identifiers with and
without a helpful column name, identifiers mixed with junk, names that promise a category the
values do not keep, and columns that hold nothing sensitive. The detector's confidence is
calibrated on this benchmark (`benchmarks/calibrate_detection.py`) and its calibration error is
tested on a second benchmark built from another seed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from faker import Faker

from oblitus.detection.reference_lists import municipalities
from oblitus.surrogates import SurrogateFactory
from oblitus.taxonomy import DataCategory

ROWS = 80
JUNK_SHARES = (0.1, 0.3, 0.6)
NEUTRAL_NAMES = ("field_a", "col_17", "x", "value", "attr_3")

_FACTORY = SurrogateFactory("pt_BR")
_C = DataCategory

# Categories the detector can read from values, with the column names that point to them.
_VALUE_CATEGORIES: dict[DataCategory, tuple[str, ...]] = {
    _C.EMAIL: ("email", "contact_email"),
    _C.CPF: ("cpf", "cpf_titular"),
    _C.CNPJ: ("cnpj", "cnpj_empresa"),
    _C.PHONE: ("phone", "telefone"),
    _C.CREDIT_CARD: ("credit_card", "card_number"),
    _C.IBAN: ("iban",),
    _C.IP_ADDRESS: ("ip_address",),
    _C.URL: ("website", "url"),
    _C.CRYPTO_WALLET: ("wallet",),
}

# Categories the detector can only read from the column name, with realistic values.
_NAME_ONLY_CATEGORIES: dict[DataCategory, tuple[str, ...]] = {
    _C.STREET_ADDRESS: ("address", "endereco"),
    _C.CITY: ("city", "cidade"),
    _C.PERSON_NAME: ("full_name", "nome_completo"),
}


@dataclass(frozen=True, slots=True)
class LabelledColumn:
    """One column of the benchmark and the category it really holds."""

    name: str
    values: pd.Series
    truth: DataCategory


def build_benchmark(seed: int = 0) -> list[LabelledColumn]:
    """Return the benchmark columns for a seed, always the same for the same seed."""
    rng = np.random.default_rng(seed)
    columns: list[LabelledColumn] = []
    for category, hints in _VALUE_CATEGORIES.items():
        columns += _identifier_columns(category, hints, rng)
    for category, hints in _NAME_ONLY_CATEGORIES.items():
        columns += _name_only_columns(category, hints, rng)
    columns += _unrelated_columns(rng)
    columns += _prose_columns(rng)
    columns += _broken_promise_columns(rng)
    columns += _hard_columns(rng)
    return columns


def _fresh_values(category: DataCategory, count: int, rng: np.random.Generator) -> list[str]:
    return _FACTORY.many(category, count, rng)


def _identifier_columns(
    category: DataCategory, hints: tuple[str, ...], rng: np.random.Generator
) -> list[LabelledColumn]:
    columns = []
    for name in (*hints, *_pick(NEUTRAL_NAMES, 2, rng)):
        columns.append(LabelledColumn(name, _series(_fresh_values(category, ROWS, rng)), category))
    for share in JUNK_SHARES:
        for name in (hints[0], str(rng.choice(NEUTRAL_NAMES))):
            values = _with_junk(_fresh_values(category, ROWS, rng), share, rng)
            columns.append(LabelledColumn(name, _series(values), category))
    return columns


def _name_only_columns(
    category: DataCategory, hints: tuple[str, ...], rng: np.random.Generator
) -> list[LabelledColumn]:
    return [
        LabelledColumn(name, _series(_fresh_values(category, ROWS, rng)), category)
        for name in hints
    ]


def _unrelated_columns(rng: np.random.Generator) -> list[LabelledColumn]:
    """Columns with no personal data, apart from business codes that single out a record."""
    columns = []
    for name in ("amount", "quantity", "score"):
        numbers = pd.Series(rng.normal(100.0, 30.0, ROWS).round(2))
        columns.append(LabelledColumn(name, numbers, _C.GENERIC_NUMERIC))
    for name in ("tier", "plan", "status"):
        labels = pd.Series(rng.choice(["gold", "silver", "bronze", "basic"], ROWS))
        columns.append(LabelledColumn(name, labels, _C.GENERIC_CATEGORY))
    for name in ("order_ref", "sku", "protocolo"):
        codes = [f"{name[:3].upper()}-{number:06d}" for number in rng.integers(0, 10**6, ROWS)]
        columns.append(LabelledColumn(name, _series(codes), _C.RECORD_ID))
    for name in ("order_number", "nsu"):
        digits = [str(number) for number in rng.integers(10**10, 10**11, ROWS)]
        columns.append(LabelledColumn(name, _series(digits), _C.RECORD_ID))
    for name in _pick(NEUTRAL_NAMES, 2, rng):
        identifiers = _fresh_values(_C.RECORD_ID, ROWS, rng)
        columns.append(LabelledColumn(name, _series(identifiers), _C.RECORD_ID))
    for name in _pick(NEUTRAL_NAMES, 2, rng):
        columns.append(LabelledColumn(name, _series(_municipality_sample(rng)), _C.CITY))
    return columns


def _prose_columns(rng: np.random.Generator) -> list[LabelledColumn]:
    columns = []
    for share in (0.0, 0.0, 0.15, 0.3, 0.5):
        texts = _fresh_values(_C.GENERIC_TEXT, ROWS, rng)
        for index in range(ROWS):
            if rng.random() < share:
                contact = _fresh_values(_C.EMAIL, 1, rng)[0]
                texts[index] = f"{texts[index]} contact {contact}"
        truth = _C.FREE_TEXT_WITH_PII if share > 0 else _C.GENERIC_TEXT
        columns.append(LabelledColumn("notes", _series(texts), truth))
    return columns


def _broken_promise_columns(rng: np.random.Generator) -> list[LabelledColumn]:
    """Columns whose name says one thing and whose values say another."""
    columns = []
    for name in ("email", "cpf", "phone", "ip_address"):
        junk = [_junk_like(8, rng) for _ in range(ROWS)]
        columns.append(LabelledColumn(name, _series(junk), _C.GENERIC_TEXT))
    for name in ("email_count", "address_type", "phone_verified"):
        counts = pd.Series(rng.integers(0, 5, ROWS))
        columns.append(LabelledColumn(name, counts, _C.GENERIC_NUMERIC))
    return columns


def _municipality_sample(rng: np.random.Generator) -> list[str]:
    names = sorted(municipalities())
    return [str(name).title() for name in rng.choice(names, ROWS)]


def _pick(options: tuple[str, ...], count: int, rng: np.random.Generator) -> list[str]:
    return [str(item) for item in rng.choice(options, count, replace=False)]


def _series(values: list[str]) -> pd.Series:
    return pd.Series(values, dtype=object)


def _with_junk(values: list[str], share: float, rng: np.random.Generator) -> list[str]:
    """Replace a share of the values by text of the same length that is not an identifier."""
    mixed = list(values)
    for index in range(len(mixed)):
        if rng.random() < share:
            mixed[index] = _junk_like(len(mixed[index]), rng)
    return mixed


def _junk_like(length: int, rng: np.random.Generator) -> str:
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    return "".join(str(letter) for letter in rng.choice(alphabet, length))


def _hard_columns(rng: np.random.Generator) -> list[LabelledColumn]:
    """Columns the rules often get wrong, so the confidence has errors to learn from."""
    columns = []
    for name in _pick(NEUTRAL_NAMES, 3, rng):
        people = _fresh_values(_C.PERSON_NAME, ROWS, rng)
        columns.append(LabelledColumn(name, _series(people), _C.PERSON_NAME))
    faker = _seeded_faker("pt_BR", rng)
    english_faker = _seeded_faker("en_US", rng)
    for name in _pick(NEUTRAL_NAMES, 2, rng):
        companies = [faker.company() for _ in range(ROWS)]
        columns.append(LabelledColumn(name, _series(companies), _C.ORGANIZATION))
    for name in _pick(NEUTRAL_NAMES, 2, rng):
        products = [" ".join(w.title() for w in faker.words(3)) for _ in range(ROWS)]
        columns.append(LabelledColumn(name, _series(products), _C.GENERIC_TEXT))
    for name in _pick(NEUTRAL_NAMES, 2, rng):
        cities = [english_faker.city() for _ in range(ROWS)]
        columns.append(LabelledColumn(name, _series(cities), _C.CITY))
    for name in _pick(NEUTRAL_NAMES, 2, rng):
        digits = [str(number) for number in rng.integers(10**8, 10**15, ROWS)]
        columns.append(LabelledColumn(name, _series(digits), _C.RECORD_ID))
    for name in _pick(NEUTRAL_NAMES, 3, rng):
        mixed = _fresh_values(_C.CPF, ROWS * 3 // 5, rng) + _fresh_values(
            _C.CNPJ, ROWS * 2 // 5, rng
        )
        columns.append(LabelledColumn(name, _series(mixed), _C.CPF))
    for share in (0.1, 0.2, 0.3, 0.4):
        texts = _fresh_values(_C.GENERIC_TEXT, ROWS, rng)
        for index in range(ROWS):
            if rng.random() < share:
                number = _fresh_values(_C.CPF, 1, rng)[0]
                texts[index] = f"{texts[index]} documento {number}"
        columns.append(LabelledColumn("observation", _series(texts), _C.FREE_TEXT_WITH_PII))
    return columns


def _seeded_faker(locale: str, rng: np.random.Generator) -> Faker:
    faker = Faker(locale)
    faker.seed_instance(int(rng.integers(0, 2**31)))
    return faker
