"""Errors raised by the library. Catch `AnonymizerError` to handle all of them."""


class AnonymizerError(Exception):
    """Base class for every error this library raises on purpose."""


class PlanError(AnonymizerError):
    """An anonymization plan names a missing column, an unknown technique or bad parameters."""


class SchemaError(AnonymizerError):
    """A dataset schema cannot produce the requested data."""


class TableFormatError(AnonymizerError):
    """A table file has an extension the library cannot read or write."""


class KeyGenerationError(AnonymizerError):
    """A key cannot be generated: an unknown entropy source or an unusable key length."""


class EntropySourceUnavailableError(AnonymizerError):
    """An entropy source cannot deliver now, such as a missing device or no network."""


class PrivacyBudgetError(AnonymizerError):
    """A differential-privacy request lacks a declared bound or domain, or would spend more
    privacy budget than was declared."""


class MissingExtraError(AnonymizerError, ImportError):
    """A feature needs an optional dependency that is not installed."""
