import pandas as pd
import pytest

from data_anonymizer.detection import scan, scan_column
from data_anonymizer.detection.column_names import category_from_name, name_tokens
from data_anonymizer.detection.patterns import (
    is_valid_cnpj,
    is_valid_cpf,
    is_valid_iban,
    passes_luhn,
)
from data_anonymizer.taxonomy import ColumnKind, DataCategory, Sensitivity


@pytest.mark.parametrize(
    ("validator", "valid", "invalid"),
    [
        (is_valid_cpf, "529.982.247-25", "529.982.247-26"),
        (is_valid_cpf, "52998224725", "11111111111"),
        (is_valid_cnpj, "11.222.333/0001-81", "11.222.333/0001-82"),
        (passes_luhn, "4111 1111 1111 1111", "4111 1111 1111 1112"),
        (is_valid_iban, "GB82WEST12345698765432", "GB82WEST12345698765433"),
    ],
)
def test_checksum_validators_accept_valid_and_reject_altered_numbers(validator, valid, invalid):
    assert validator(valid)
    assert not validator(invalid)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("email", DataCategory.EMAIL),
        ("dataNascimento", DataCategory.DATE_OF_BIRTH),
        ("Endereço", DataCategory.STREET_ADDRESS),
        ("zip_code", DataCategory.POSTAL_CODE),
        ("customer_id", DataCategory.RECORD_ID),
        ("name", DataCategory.PERSON_NAME),
        ("product_name", None),
        ("zipper_size", None),
    ],
)
def test_column_names_suggest_categories(column, expected):
    assert category_from_name(column) == expected


def test_name_tokens_split_camel_case_and_strip_accents():
    assert name_tokens("dataDeNascimento") == ["data", "de", "nascimento"]
    assert name_tokens("Endereço Residencial") == ["endereco", "residencial"]


def test_scan_classifies_every_column_of_a_customer_table(people):
    findings = {finding.column: finding for finding in scan(people)}

    expected = {
        "customer_id": DataCategory.RECORD_ID,
        "full_name": DataCategory.PERSON_NAME,
        "email": DataCategory.EMAIL,
        "cpf": DataCategory.CPF,
        "phone": DataCategory.PHONE,
        "birth_date": DataCategory.DATE_OF_BIRTH,
        "age": DataCategory.AGE,
        "cep": DataCategory.POSTAL_CODE,
        "city": DataCategory.CITY,
        "gender": DataCategory.GENDER,
        "salary": DataCategory.INCOME,
        "diagnosis": DataCategory.HEALTH,
        "notes": DataCategory.FREE_TEXT_WITH_PII,
        "basket_value": DataCategory.GENERIC_NUMERIC,
    }
    assert {column: finding.category for column, finding in findings.items()} == expected
    assert findings["basket_value"].sensitivity is Sensitivity.NON_SENSITIVE


def test_values_identify_a_column_whose_name_says_nothing():
    values = pd.Series(["529.982.247-25", "111.444.777-35", "390.533.447-05"] * 10)

    finding = scan_column("col_7", values)

    assert finding.category is DataCategory.CPF
    assert finding.confidence == 1.0


def test_name_contradicted_by_values_lowers_confidence():
    finding = scan_column("email", pd.Series(["alpha", "beta", "gamma"] * 10))

    assert finding.category is DataCategory.EMAIL
    assert finding.confidence < 0.6
    assert "do not match" in finding.evidence


def test_unique_short_codes_are_flagged_as_identifiers():
    codes = pd.Series([f"ORD{index:05d}" for index in range(100)])

    finding = scan_column("reference", codes)

    assert finding.kind is ColumnKind.TEXT
    assert finding.category is DataCategory.RECORD_ID


def test_dates_stored_as_text_are_recognized_as_dates():
    finding = scan_column("created", pd.Series(["2024-01-05", "2024-02-11", "2023-12-30"] * 5))

    assert finding.kind is ColumnKind.DATETIME
    assert finding.category is DataCategory.GENERIC_DATETIME
