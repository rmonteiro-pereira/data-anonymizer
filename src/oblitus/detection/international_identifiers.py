"""Identifiers of other countries, checked with the optional `python-stdnum` and `phonenumbers`.

The built-in rules cover Brazil and a handful of other countries. When the `identifiers` extra
is installed (`pip install "oblitus[identifiers]"`), the detector also recognizes more national
numbers and international phone numbers. Without it, nothing here runs and the built-in rules
decide alone. Both packages are imported when first needed, never when the library is imported.
"""

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from types import ModuleType

from oblitus.detection.identifier_boundaries import bounded


@dataclass(frozen=True, slots=True)
class StdnumIdentifier:
    """A national number: how it looks, and the `stdnum` module that checks it."""

    label: str
    pattern: str
    module: str


# Only formats with punctuation, letters or a check that random digits rarely pass are listed,
# because bare digit strings would match order numbers and timestamps.
STDNUM_IDENTIFIERS: tuple[StdnumIdentifier, ...] = (
    StdnumIdentifier(
        "Mexican CURP", bounded(r"[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d", ignore_case=True), "mx.curp"
    ),
    StdnumIdentifier("Chilean RUT", bounded(r"\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]"), "cl.rut"),
    StdnumIdentifier("Argentine CUIT/CUIL", bounded(r"\d{2}-\d{8}-\d"), "ar.cuit"),
    StdnumIdentifier("Swedish personnummer", bounded(r"\d{6}[-+]\d{4}"), "se.personnummer"),
    StdnumIdentifier("Danish CPR", bounded(r"\d{6}-\d{4}"), "dk.cpr"),
    StdnumIdentifier(
        "Finnish HETU", bounded(r"\d{6}[-+A]\d{3}[0-9A-Y]", ignore_case=True), "fi.hetu"
    ),
    StdnumIdentifier("Irish PPS number", bounded(r"\d{7}[A-W][A-IW]?", ignore_case=True), "ie.pps"),
    StdnumIdentifier(
        "Belgian national number", bounded(r"\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}"), "be.nn"
    ),
    StdnumIdentifier("Swiss AHV number", bounded(r"756\.\d{4}\.\d{4}\.\d{2}"), "ch.ssn"),
    StdnumIdentifier(
        "French NIR",
        bounded(r"[12]\s?\d{2}\s?\d{2}\s?(?:\d{2}|2[AB])\s?\d{3}\s?\d{3}\s?\d{2}"),
        "fr.nir",
    ),
    StdnumIdentifier("Indian PAN", bounded(r"[A-Z]{5}\d{4}[A-Z]", ignore_case=True), "in_.pan"),
)


def stdnum_is_installed() -> bool:
    """True when `python-stdnum` can be imported."""
    return importlib.util.find_spec("stdnum") is not None


def stdnum_validator(module: str) -> Callable[[str], bool]:
    """Return a validator that asks one `stdnum` module whether a value is a valid number."""

    def is_valid(value: str) -> bool:
        checker = importlib.import_module(f"stdnum.{module}")
        return bool(checker.is_valid(value))

    return is_valid


def is_valid_international_phone(value: str) -> bool:
    """True when `phonenumbers` is installed and says a `+` number is valid for its country."""
    if not value.startswith("+"):
        return False
    library = _phonenumbers()
    if library is None:
        return False
    try:
        return bool(library.is_valid_number(library.parse(value)))
    except library.NumberParseException:
        return False


@cache
def _phonenumbers() -> ModuleType | None:
    if importlib.util.find_spec("phonenumbers") is None:
        return None
    return importlib.import_module("phonenumbers")
