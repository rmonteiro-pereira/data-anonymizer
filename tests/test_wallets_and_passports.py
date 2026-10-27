import pandas as pd
import pytest

from oblitus import Anonymizer
from oblitus.detection import scan_column
from oblitus.detection.crypto_wallets import (
    eip55_checksum_case,
    is_valid_bitcoin_address,
    is_valid_ethereum_address,
    keccak256,
)
from oblitus.detection.passport_mrz import is_valid_passport_mrz, weighted_check_digit
from oblitus.detection.patterns import TEXT_RECOGNIZERS
from oblitus.detection.text_entities import find_entities
from oblitus.surrogates import SurrogateFactory
from oblitus.taxonomy import DataCategory

MRZ_LINE_ONE = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
MRZ_LINE_TWO = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
MRZ = f"{MRZ_LINE_ONE}\n{MRZ_LINE_TWO}"

LEGACY_BITCOIN = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
SEGWIT_BITCOIN = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
TAPROOT_BITCOIN = "bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0"
CHECKSUMMED_ETHEREUM = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"


def test_keccak256_matches_the_known_empty_digest():
    assert keccak256(b"").hex() == (
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )


@pytest.mark.parametrize("address", [LEGACY_BITCOIN, SEGWIT_BITCOIN, TAPROOT_BITCOIN])
def test_valid_bitcoin_addresses_pass(address):
    assert is_valid_bitcoin_address(address)


@pytest.mark.parametrize(
    "address",
    [
        LEGACY_BITCOIN[:-1] + "3",
        SEGWIT_BITCOIN[:-1] + "5",
        SEGWIT_BITCOIN[:10].upper() + SEGWIT_BITCOIN[10:],
        "1" + "a" * 30,
    ],
)
def test_bitcoin_addresses_with_a_bad_checksum_fail(address):
    assert not is_valid_bitcoin_address(address)


def test_ethereum_checksum_applies_only_to_mixed_case():
    lowercase = CHECKSUMMED_ETHEREUM.lower()
    wrong_case = CHECKSUMMED_ETHEREUM[:-1] + "D"

    assert is_valid_ethereum_address(CHECKSUMMED_ETHEREUM)
    assert is_valid_ethereum_address(lowercase)
    assert is_valid_ethereum_address("0x" + lowercase[2:].upper())
    assert not is_valid_ethereum_address(wrong_case)
    assert eip55_checksum_case(lowercase[2:]) == CHECKSUMMED_ETHEREUM[2:]


def test_a_wallet_column_is_found_and_wallets_in_text_are_redacted():
    wallets = pd.Series([LEGACY_BITCOIN, SEGWIT_BITCOIN, CHECKSUMMED_ETHEREUM] * 10)

    finding = scan_column("payout", wallets)

    assert finding.category is DataCategory.CRYPTO_WALLET
    assert finding.sensitivity.value == "direct_identifier"
    text = f"send to {LEGACY_BITCOIN} or {CHECKSUMMED_ETHEREUM}"
    spans = find_entities(text, TEXT_RECOGNIZERS)
    assert [span.category for span in spans] == [DataCategory.CRYPTO_WALLET] * 2


def test_fake_wallets_are_valid_and_consistent():
    factory = SurrogateFactory()

    fakes = [factory.value_for(DataCategory.CRYPTO_WALLET, seed) for seed in range(20)]

    assert fakes == [factory.value_for(DataCategory.CRYPTO_WALLET, seed) for seed in range(20)]
    assert all(is_valid_bitcoin_address(fake) or is_valid_ethereum_address(fake) for fake in fakes)


def test_the_icao_specimen_passport_zone_is_valid():
    assert weighted_check_digit("L898902C3") == 6
    assert is_valid_passport_mrz(MRZ)
    assert is_valid_passport_mrz(MRZ.replace("\n", ""))


@pytest.mark.parametrize(
    "broken",
    [
        MRZ.replace("7408122", "7408123"),
        MRZ.replace("L898902C36", "L898902C37"),
        MRZ.replace("<<<<<10", "<<<<<11"),
    ],
)
def test_a_passport_zone_with_a_wrong_check_digit_is_rejected(broken):
    assert not is_valid_passport_mrz(broken)


def test_a_passport_zone_is_found_in_a_column_and_in_text():
    column = pd.Series([MRZ] * 30)

    finding = scan_column("travel_document", column)

    assert finding.category is DataCategory.NATIONAL_ID
    spans = find_entities(f"scan: {MRZ} end", TEXT_RECOGNIZERS)
    assert [span.category for span in spans] == [DataCategory.NATIONAL_ID]


def test_wallet_columns_get_an_action_in_the_proposed_plan():
    frame = pd.DataFrame({"wallet": [LEGACY_BITCOIN, CHECKSUMMED_ETHEREUM] * 20})
    anonymizer = Anonymizer(seed=1)

    plan = anonymizer.propose(frame)

    assert plan.action_for("wallet") is not None


def _found_categories(text: str) -> list[DataCategory]:
    return [span.category for span in find_entities(text, TEXT_RECOGNIZERS)]


@pytest.mark.parametrize(
    "text",
    [
        f"Beneficiary wallet_{CHECKSUMMED_ETHEREUM.lower()} approved",
        f"key_{CHECKSUMMED_ETHEREUM}",
        f"{CHECKSUMMED_ETHEREUM}_backup",
        f"payout_{LEGACY_BITCOIN}_old",
        "cpf_52998224725 ok",
        "cartao_4111111111111111 ok",
        "mail me: owner_x@example.org",
        "tel_11987654321 ok",
    ],
)
def test_identifiers_tagged_with_underscores_are_found_in_text(text):
    assert _found_categories(text)


def test_identifiers_glued_to_letters_are_still_not_found():
    assert not _found_categories(f"abc{CHECKSUMMED_ETHEREUM.lower()}x")


def test_a_wallet_with_an_underscore_tag_is_redacted_in_the_proposed_release():
    wallet = CHECKSUMMED_ETHEREUM.lower()
    notes = [f"Beneficiary wallet_{wallet} approved", f"Beneficiary wallet_{wallet[:-4]}ffff ok"]
    frame = pd.DataFrame({"note": notes * 10, "diagnosis": ["D", "H"] * 10})
    anonymizer = Anonymizer(seed=1)

    result = anonymizer.apply(frame, anonymizer.propose(frame))

    assert not result.data["note"].str.contains("0x").any()


def test_a_lowercase_passport_zone_is_found_and_validated():
    lowered = MRZ.lower()

    assert is_valid_passport_mrz(lowered)
    assert _found_categories(f"scan: {lowered} end") == [DataCategory.NATIONAL_ID]
    finding = scan_column("travel_document", pd.Series([lowered] * 30))
    assert finding.category is DataCategory.NATIONAL_ID


def test_a_lowercase_passport_zone_with_a_broken_check_digit_is_rejected():
    assert not is_valid_passport_mrz(MRZ.lower().replace("7408122", "7408123"))
