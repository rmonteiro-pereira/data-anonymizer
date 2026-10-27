import time

import pandas as pd
import pytest

from data_anonymizer.detection import checksums, scan, scan_column
from data_anonymizer.detection.column_names import category_from_name, name_tokens
from data_anonymizer.detection.patterns import (
    RECOGNIZERS,
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


@pytest.mark.parametrize(
    ("validator", "valid", "invalid"),
    [
        (checksums.is_valid_cnpj, "12.ABC.345/01DE-35", "12.ABC.345/01DE-36"),
        (checksums.is_valid_spanish_id, "12345678Z", "12345678A"),
        (checksums.is_valid_spanish_id, "X1234567L", "X1234567M"),
        (checksums.is_valid_portuguese_nif, "123456789", "123456788"),
        (checksums.is_valid_dutch_bsn, "111222333", "111222334"),
        (checksums.is_valid_fiscal_code, "RSSMRA85T10A562S", "RSSMRA85T10A562T"),
        (checksums.is_valid_nino, "AB123456C", "DA123456C"),
        (checksums.is_valid_imei, "490154203237518", "490154203237519"),
        (checksums.is_valid_aadhaar, "234123412346", "234123412347"),
        (checksums.is_valid_german_tax_id, "36574261809", "36574261808"),
        (checksums.is_ip_address, "2001:db8::8a2e:370:7334", "12:30:00"),
    ],
)
def test_international_and_new_brazilian_checksums(validator, valid, invalid):
    assert validator(valid)
    assert not validator(invalid)


@pytest.mark.parametrize(
    "validator",
    [
        checksums.is_valid_cnh,
        checksums.is_valid_nis,
        checksums.is_valid_renavam,
        checksums.is_valid_voter_id,
        checksums.is_valid_cns,
        checksums.is_valid_invoice_key,
    ],
)
def test_changing_any_digit_breaks_a_brazilian_document_number(validator):
    length = {
        checksums.is_valid_cns: 15,
        checksums.is_valid_invoice_key: 44,
        checksums.is_valid_voter_id: 12,
    }.get(validator, 11)
    valid = _first_valid(validator, length)
    altered = [
        valid[:position] + str((int(valid[position]) + step) % 10) + valid[position + 1 :]
        for position in range(length)
        for step in range(1, 10)
    ]

    # Mod-11 schemes miss a few changes by design, such as a digit whose weight is 11.
    rejected = sum(not validator(number) for number in altered) / len(altered)
    assert rejected > 0.8


def _first_valid(validator, length: int) -> str:
    seed = 7
    for _ in range(200_000):
        seed = (seed * 6364136223846793005 + 1442695040888963407) % 2**64
        prefix = "7" if validator is checksums.is_valid_cns else ""
        digits = (prefix + str(seed) * 6)[:length]
        if validator is checksums.is_valid_voter_id:
            digits = digits[:8] + "12" + digits[10:]
        if validator(digits):
            return digits
    raise AssertionError("no valid number found")


def test_timestamps_and_order_numbers_are_not_phones_or_documents():
    epoch_seconds = pd.Series(range(1_700_000_000, 1_700_000_000 + 50 * 3_600, 3_600))
    order_numbers = pd.Series([f"{index:012d}" for index in range(10**11, 10**11 + 50)])

    assert scan_column("event_time", epoch_seconds).category is not DataCategory.PHONE
    assert scan_column("order_number", order_numbers).category not in {
        DataCategory.PHONE,
        DataCategory.CPF,
        DataCategory.CREDIT_CARD,
    }


def test_bare_brazilian_mobiles_are_phones():
    mobiles = pd.Series([f"119{index:08d}" for index in range(40)])

    assert scan_column("contato", mobiles).category is DataCategory.PHONE


def _with_luhn_digit(prefix: str) -> str:
    return next(
        prefix + str(digit) for digit in range(10) if checksums.passes_luhn(prefix + str(digit))
    )


def test_imeis_are_not_credit_cards():
    imeis = pd.Series([_with_luhn_digit(f"35{index:012d}") for index in range(30)])

    assert scan_column("device", imeis).category is DataCategory.IMEI


def test_integer_ids_with_lost_leading_zeros_are_still_recognized():
    cpfs = pd.Series([int("01234567890"), int("00000000191")] * 15)

    assert scan_column("documento", cpfs).category is DataCategory.CPF


def test_name_hint_formats_need_their_name():
    uuids = pd.Series(["123e4567-e89b-42d3-a456-426614174000"] * 25)

    assert scan_column("chave_pix", uuids).category is DataCategory.PIX_KEY
    assert scan_column("session", uuids).category is not DataCategory.PIX_KEY


def test_birth_dates_are_recognized_from_their_values():
    dates = pd.Series(pd.date_range("1950-01-01", "2005-01-01", periods=200).normalize())

    finding = scan_column("dt", dates)

    assert finding.category is DataCategory.DATE_OF_BIRTH
    assert finding.confidence < 0.6


def _recognizer(category: DataCategory, label: str):
    return next(
        recognizer
        for recognizer in RECOGNIZERS
        if recognizer.category is category and recognizer.label == label
    )


@pytest.mark.parametrize(
    "number",
    [
        "555-123-4567",
        "(555) 123-4567",
        "555.123.4567",
        "(555)123-4567",
        "+1 555 123 4567",
        "+1 (555) 123-4567",
        "+1-555-123-4567",
        "1-800-555-0199",
        "+14155552671",
        "+5511987654321",
        "55 11 98765 4321",
        "(11) 98765-4321",
        "(11) 3333-4444",
        "+44 7911 123456",
        "07911 123456",
        "+33 1 23 45 67 89",
        "01 23 45 67 89",
        "+34 612 345 678",
        "+351 912 345 678",
        "+39 333 123 4567",
        "+91 98765 43210",
        "0412 345 678",
        "+54 9 11 1234-5678",
    ],
)
def test_phone_numbers_in_real_world_layouts_are_recognized(number):
    assert _recognizer(DataCategory.PHONE, "phone number").matches(number)


@pytest.mark.parametrize(
    "value",
    [
        "529.982.247-25",
        "11.222.333/0001-81",
        "192.168.100.200",
        "10.20.30.40",
        "4.2.17.203",
        "2024-01-05",
        "05/01/2024",
        "2024-01-05 10:30",
        "12:30:45",
        "1234567",
        "5551234567",
    ],
)
def test_documents_addresses_versions_and_dates_are_not_phones(value):
    assert not _recognizer(DataCategory.PHONE, "phone number").matches(value)


def test_unnamed_us_phone_columns_are_phones_and_prose_with_a_phone_is_free_text():
    numbers = pd.Series([f"({200 + index % 700}) 555-{1000 + index:04d}" for index in range(300)])
    prose = pd.Series([f"Call me back on 555-123-{1000 + index} please" for index in range(100)])

    assert scan_column("col_a", numbers).category is DataCategory.PHONE
    assert scan_column("notes", prose).category is DataCategory.FREE_TEXT_WITH_PII


def _luhn_number(prefix: str, length: int) -> str:
    return _with_luhn_digit(prefix + "1" * (length - len(prefix) - 1))


@pytest.mark.parametrize(
    ("prefix", "length", "accepted"),
    [
        ("4", 13, True),
        ("4", 16, True),
        ("4", 19, True),
        ("4", 14, False),
        ("4", 17, False),
        ("51", 16, True),
        ("2221", 16, True),
        ("51", 13, False),
        ("51", 14, False),
        ("51", 19, False),
        ("34", 15, True),
        ("34", 16, False),
        ("36", 14, True),
        ("36", 19, True),
        ("36", 13, False),
        ("6011", 16, True),
        ("6011", 19, True),
        ("62", 17, True),
        ("6062", 16, True),
        ("636368", 16, True),
        ("3528", 16, True),
        ("3528", 19, True),
        ("31", 16, False),
        ("60", 13, False),
    ],
)
def test_card_numbers_must_have_a_length_their_network_issues(prefix, length, accepted):
    assert checksums.is_valid_card(_luhn_number(prefix, length)) is accepted


@pytest.mark.parametrize(
    ("category", "label", "value"),
    [
        (DataCategory.IBAN, "IBAN", "GB82WEST12345698765432"),
        (DataCategory.NATIONAL_ID, "Spanish DNI/NIE", "12345678Z"),
        (DataCategory.NATIONAL_ID, "UK National Insurance number", "AB123456C"),
        (DataCategory.NATIONAL_ID, "Italian codice fiscale", "RSSMRA85T10A562S"),
        (DataCategory.CNPJ, "CNPJ", "12.ABC.345/01DE-35"),
        (DataCategory.VEHICLE_PLATE, "vehicle plate", "ABC1D23"),
    ],
)
def test_identifiers_with_letters_are_recognized_in_lower_case(category, label, value):
    recognizer = _recognizer(category, label)

    assert recognizer.matches(value)
    assert recognizer.matches(value.lower())


def test_email_search_in_a_long_cell_takes_linear_time():
    email = _recognizer(DataCategory.EMAIL, "email")
    blob = "Ab3dEf9x" * 25_000 + " end"

    started = time.perf_counter()
    spans = email.find_spans(blob + " write to ana.silva+x@mail.example.com")
    elapsed = time.perf_counter() - started

    assert len(spans) == 1
    assert elapsed < 0.5


@pytest.mark.parametrize(
    "text",
    ["Module::Class failed", "Score 3::4", "std::vector", "a::b", "::", "12:30:45", "fwd:: x"],
)
def test_colons_in_prose_are_not_ipv6_addresses(text):
    assert _recognizer(DataCategory.IP_ADDRESS, "IPv6 address").find_spans(text) == []


def test_ipv6_addresses_with_three_groups_are_found_in_text():
    spans = _recognizer(DataCategory.IP_ADDRESS, "IPv6 address").find_spans(
        "login from 2001:db8::8a2e:370:7334 ok"
    )

    assert spans == [(11, 34)]


def test_log_lines_and_order_numbers_in_prose_are_not_free_text_with_pii():
    log_lines = pd.Series([f"Module::Class failed with code {index}" for index in range(100)])
    imei_like_orders = pd.Series(
        [f"Pedido {_with_luhn_digit(f'55{index:012d}')} entregue" for index in range(100)]
    )

    assert scan_column("notes", log_lines).category is not DataCategory.FREE_TEXT_WITH_PII
    assert scan_column("notes", imei_like_orders).category is not DataCategory.FREE_TEXT_WITH_PII


def test_embedded_hits_must_beat_the_chance_rate_of_the_recognizers_that_fired():
    bare_mobile_lookalike = "11987654321"
    cells = [
        f"Pedido {bare_mobile_lookalike if index % 8 == 0 else '20230100000'} entregue"
        for index in range(200)
    ]

    assert scan_column("notes", pd.Series(cells)).category is not DataCategory.FREE_TEXT_WITH_PII
