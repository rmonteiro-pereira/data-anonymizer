"""The built-in techniques, each described by its family, its effect on statistics and the kinds
of column it accepts."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pandas as pd

from data_anonymizer.taxonomy import NUMERIC_KINDS, ColumnKind
from data_anonymizer.techniques import encryption, generalization, masking, noise, perturbation
from data_anonymizer.techniques import pseudonymization as pseudo
from data_anonymizer.techniques.context import TechniqueContext

TechniqueFunction = Callable[..., pd.Series]


class TechniqueFamily(StrEnum):
    """The broad approach a technique takes."""

    KEEP = "keep"
    MASKING = "masking"
    PSEUDONYMIZATION = "pseudonymization"
    ENCRYPTION = "encryption"
    GENERALIZATION = "generalization"
    NOISE = "noise"
    PERTURBATION = "perturbation"


class StatisticalEffect(StrEnum):
    """What a technique does to the statistics an analyst would compute on the column.

    `preserves_relationships`: distinct counts, joins and links to other columns survive.
    `preserves_distribution`: the column's own distribution survives, roughly or exactly,
    but links to other columns may weaken.
    `distorts`: the distribution changes; analysis needs to account for it.
    """

    PRESERVES_RELATIONSHIPS = "preserves_relationships"
    PRESERVES_DISTRIBUTION = "preserves_distribution"
    DISTORTS = "distorts"


@dataclass(frozen=True, slots=True)
class Technique:
    """One way of transforming a column, with what it needs and what it costs."""

    name: str
    family: TechniqueFamily
    effect: StatisticalEffect
    kinds: frozenset[ColumnKind]
    function: TechniqueFunction
    summary: str
    inverse: TechniqueFunction | None = None
    linkable: bool = False
    unlinking_flag: str | None = None

    @property
    def reversible(self) -> bool:
        """True when the secret key's holder can recover the original values."""
        return self.inverse is not None

    def keeps_link(self, params: Mapping[str, Any]) -> bool:
        """True when the output can still be tied back to the person.

        Reversible techniques always keep the link. Linkable techniques give the same
        output for the same person, or let anyone test a guessed value against the
        output, unless their `unlinking_flag` parameter is set to False.
        """
        if self.reversible:
            return True
        if not self.linkable:
            return False
        if self.unlinking_flag is None:
            return True
        return bool(params.get(self.unlinking_flag, True))

    def parameter_names(self) -> list[str]:
        """Return the keyword parameters the technique accepts."""
        signature = inspect.signature(self.function)
        return [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        ]

    def check_parameters(self, params: Mapping[str, Any]) -> None:
        """Raise TypeError when params name something the technique does not accept."""
        signature = inspect.signature(self.function)
        signature.bind(None, None, **params)
        if self.inverse is not None:
            inspect.signature(self.inverse).bind(None, None, **params)

    def apply(
        self, values: pd.Series, context: TechniqueContext, params: Mapping[str, Any]
    ) -> pd.Series:
        """Transform a column and return the new column."""
        return self.function(values, context, **params)

    def reverse(
        self, values: pd.Series, context: TechniqueContext, params: Mapping[str, Any]
    ) -> pd.Series:
        """Recover the original column from a transformed one."""
        if self.inverse is None:
            raise ValueError(f"{self.name} cannot be reversed")
        return self.inverse(values, context, **params)


_ALL_KINDS = frozenset(ColumnKind)
_NUMERIC = frozenset(NUMERIC_KINDS)
_DATES = frozenset({ColumnKind.DATETIME})
_TEXT_LIKE = frozenset({ColumnKind.TEXT, ColumnKind.CATEGORY, ColumnKind.INTEGER})
_CATEGORICAL = frozenset({ColumnKind.CATEGORY, ColumnKind.BOOLEAN})

_KEEP = TechniqueFamily.KEEP
_MASKING = TechniqueFamily.MASKING
_PSEUDO = TechniqueFamily.PSEUDONYMIZATION
_ENCRYPT = TechniqueFamily.ENCRYPTION
_GENERAL = TechniqueFamily.GENERALIZATION
_NOISE = TechniqueFamily.NOISE
_PERTURB = TechniqueFamily.PERTURBATION

_RELATIONSHIPS = StatisticalEffect.PRESERVES_RELATIONSHIPS
_DISTRIBUTION = StatisticalEffect.PRESERVES_DISTRIBUTION
_DISTORTS = StatisticalEffect.DISTORTS

_BUILTINS = (
    Technique("keep", _KEEP, _RELATIONSHIPS, _ALL_KINDS, masking.keep, "Leave unchanged."),
    Technique(
        "redact", _MASKING, _DISTORTS, _ALL_KINDS, masking.redact, "Replace with a constant."
    ),
    Technique(
        "suppress", _MASKING, _DISTORTS, _ALL_KINDS, masking.suppress, "Replace with missing."
    ),
    Technique(
        "partial_mask",
        _MASKING,
        _DISTORTS,
        _TEXT_LIKE,
        masking.partial_mask,
        "Hide all but the first or last characters, keeping separators.",
    ),
    Technique(
        "redact_entities",
        _MASKING,
        _DISTORTS,
        frozenset({ColumnKind.TEXT, ColumnKind.CATEGORY}),
        masking.redact_entities,
        "Replace emails, documents and phones inside free text with labels.",
    ),
    Technique(
        "keyed_hash",
        _PSEUDO,
        _RELATIONSHIPS,
        _ALL_KINDS,
        pseudo.keyed_hash,
        "Replace with a keyed HMAC digest (SHA-2, SHA-3 or BLAKE2); consistent per key.",
        linkable=True,
    ),
    Technique(
        "scramble_characters",
        _PSEUDO,
        _RELATIONSHIPS,
        _TEXT_LIKE,
        pseudo.scramble_characters,
        "Replace digits with digits and letters with letters, keeping the format.",
        linkable=True,
    ),
    Technique(
        "surrogate",
        _PSEUDO,
        _RELATIONSHIPS,
        _TEXT_LIKE,
        pseudo.surrogate,
        "Replace with a realistic fake; consistent per value, or random per cell.",
        linkable=True,
        unlinking_flag="consistent",
    ),
    Technique(
        "argon2_hash",
        _PSEUDO,
        _RELATIONSHIPS,
        _ALL_KINDS,
        pseudo.argon2_hash,
        "Replace with a slow, memory-hard Argon2id hash; a keyed salt keeps joins.",
        linkable=True,
    ),
    Technique(
        "bcrypt_hash",
        _PSEUDO,
        _RELATIONSHIPS,
        _ALL_KINDS,
        pseudo.bcrypt_hash,
        "Replace with a slow bcrypt hash; a keyed salt keeps joins.",
        linkable=True,
    ),
    Technique(
        "encrypt",
        _ENCRYPT,
        _RELATIONSHIPS,
        _ALL_KINDS,
        encryption.encrypt,
        "Reversible AES-SIV token (deterministic) or AES-GCM token (randomized).",
        inverse=encryption.decrypt,
    ),
    Technique(
        "format_preserving_encrypt",
        _ENCRYPT,
        _RELATIONSHIPS,
        _TEXT_LIKE,
        encryption.format_preserving_encrypt,
        "Reversible NIST FF1 encryption that keeps length, format and separators.",
        inverse=encryption.format_preserving_decrypt,
    ),
    Technique(
        "bin_numeric",
        _GENERAL,
        _DISTORTS,
        _NUMERIC,
        generalization.bin_numeric,
        "Replace numbers with interval labels.",
    ),
    Technique(
        "round_numeric",
        _GENERAL,
        _DISTRIBUTION,
        _NUMERIC,
        generalization.round_numeric,
        "Round to decimals or to a multiple of a base.",
    ),
    Technique(
        "truncate_date",
        _GENERAL,
        _DISTORTS,
        _DATES,
        generalization.truncate_date,
        "Cut dates to the start of their year, month, week or day.",
    ),
    Technique(
        "truncate_string",
        _GENERAL,
        _DISTORTS,
        _TEXT_LIKE,
        generalization.truncate_string,
        "Keep only the first characters, such as a postal code's region.",
    ),
    Technique(
        "top_bottom_code",
        _GENERAL,
        _DISTRIBUTION,
        _NUMERIC,
        generalization.top_bottom_code,
        "Clip outliers to quantile or fixed limits.",
    ),
    Technique(
        "suppress_rare",
        _GENERAL,
        _DISTORTS,
        _TEXT_LIKE | _CATEGORICAL,
        generalization.suppress_rare,
        "Replace categories seen fewer than a minimum number of times.",
    ),
    Technique(
        "gaussian_noise",
        _NOISE,
        _DISTRIBUTION,
        _NUMERIC,
        noise.gaussian_noise,
        "Add normal noise, rescaled to keep the mean and standard deviation.",
    ),
    Technique(
        "laplace_noise",
        _NOISE,
        _DISTORTS,
        _NUMERIC,
        noise.laplace_noise,
        "Clip and add Laplace noise; local differential privacy with fixed bounds.",
    ),
    Technique(
        "multiplicative_noise",
        _NOISE,
        _DISTRIBUTION,
        _NUMERIC,
        noise.multiplicative_noise,
        "Multiply by a random factor near 1; suits amounts.",
    ),
    Technique(
        "date_shift",
        _NOISE,
        _DISTRIBUTION,
        _DATES,
        noise.date_shift,
        "Move each date by a random number of days.",
    ),
    Technique(
        "shuffle",
        _PERTURB,
        _DISTRIBUTION,
        _ALL_KINDS,
        perturbation.shuffle,
        "Reorder values among rows; exact distribution, no link to other columns.",
    ),
    Technique(
        "rank_swap",
        _PERTURB,
        _DISTRIBUTION,
        _NUMERIC | _DATES,
        perturbation.rank_swap,
        "Swap values with near neighbours in rank; exact distribution.",
    ),
    Technique(
        "microaggregate",
        _PERTURB,
        _DISTRIBUTION,
        _NUMERIC,
        perturbation.microaggregate,
        "Replace values with the mean of groups of at least k; exact mean.",
    ),
    Technique(
        "randomized_response",
        _PERTURB,
        _DISTORTS,
        _CATEGORICAL,
        perturbation.randomized_response,
        "Randomly replace some categories; deniable, local differential privacy.",
    ),
)

BUILTIN_TECHNIQUES: Mapping[str, Technique] = MappingProxyType(
    {technique.name: technique for technique in _BUILTINS}
)
