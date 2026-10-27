import pandas as pd
import pytest

from oblitus.detection import international_identifiers, scan_column
from oblitus.detection.checksums import is_phone_number
from oblitus.detection.patterns import RECOGNIZERS
from oblitus.taxonomy import DataCategory

NATIONAL_NUMBERS = {
    "Mexican CURP": "BADD110313HCMLNS06",
    "Chilean RUT": "76086428-5",
    "Argentine CUIT/CUIL": "20-05536168-2",
    "Swedish personnummer": "811228-9874",
    "Danish CPR": "211062-5629",
    "Finnish HETU": "131052-308T",
    "Irish PPS number": "6433435F",
    "Belgian national number": "85.07.30-033.28",
    "Swiss AHV number": "756.1234.5678.97",
    "French NIR": "2 55 08 14 168 025 38",
}

# The Danish CPR dropped its check digit in 2007, so only its structure is verified.
NUMBERS_WITHOUT_CHECK_DIGIT = {"Danish CPR"}


@pytest.fixture
def stdnum_recognizers():
    pytest.importorskip("stdnum")
    return {recognizer.label: recognizer for recognizer in RECOGNIZERS}


@pytest.mark.parametrize("label", sorted(NATIONAL_NUMBERS))
def test_national_numbers_of_other_countries_are_recognized(stdnum_recognizers, label):
    recognizer = stdnum_recognizers[label]
    value = NATIONAL_NUMBERS[label]

    assert recognizer.category is DataCategory.NATIONAL_ID
    assert recognizer.matches(value)
    if label not in NUMBERS_WITHOUT_CHECK_DIGIT:
        assert not recognizer.matches(value[:-1] + ("0" if value[-1] != "0" else "1"))


def test_a_column_of_chilean_ruts_is_a_national_id(stdnum_recognizers):
    ruts = pd.Series(["76086428-5", "12.345.678-5", "11.111.111-1", "7.654.321-6"] * 10)

    finding = scan_column("document", ruts)

    assert finding.category is DataCategory.NATIONAL_ID


def test_without_stdnum_the_extra_recognizers_are_not_registered(monkeypatch):
    monkeypatch.setattr(international_identifiers, "stdnum_is_installed", lambda: False)
    from oblitus.detection import patterns

    monkeypatch.setattr(patterns, "stdnum_is_installed", lambda: False)

    assert patterns._stdnum_recognizers() == ()


def test_international_phones_are_validated_by_phonenumbers_when_installed():
    phonenumbers = pytest.importorskip("phonenumbers")
    example = phonenumbers.example_number("IS")
    icelandic = phonenumbers.format_number(example, phonenumbers.PhoneNumberFormat.E164)

    assert is_phone_number(icelandic)
    assert not is_phone_number("+354 123")


def test_without_phonenumbers_international_phones_fall_back_to_the_digit_rules(monkeypatch):
    monkeypatch.setattr(international_identifiers, "_phonenumbers", lambda: None)

    assert not international_identifiers.is_valid_international_phone("+354 555 1234")
    assert is_phone_number("+44 7911 123456")
