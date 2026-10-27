"""The built-in techniques, each described by its family, its effect on statistics and the kinds
of column it accepts."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import narwhals as nw
import pandas as pd

from data_anonymizer.table import Table
from data_anonymizer.taxonomy import NUMERIC_KINDS, ColumnKind
from data_anonymizer.techniques import (
    encryption,
    generalization,
    masking,
    noise,
    perturbation,
    rescaling,
)
from data_anonymizer.techniques import pseudonymization as pseudo
from data_anonymizer.techniques.context import TechniqueContext

# A technique takes the table, the column's name, the run's context and its keyword
# parameters, and returns an expression for the column's new values.
TechniqueFunction = Callable[..., nw.Expr]


class TechniqueFamily(StrEnum):
    """The broad approach a technique takes."""

    KEEP = "keep"
    MASKING = "masking"
    PSEUDONYMIZATION = "pseudonymization"
    ENCRYPTION = "encryption"
    RESCALING = "rescaling"
    GENERALIZATION = "generalization"
    NOISE = "noise"
    PERTURBATION = "perturbation"


class StatisticalEffect(StrEnum):
    """What a technique does to the statistics an analyst would compute on the column.

    `preserves_relationships`: distinct counts, joins and links to other columns survive.
    `preserves_distribution`: the column's own distribution survives, roughly or exactly,
    but links to other columns may weaken.
    `preserves_shape`: the distribution's shape and the column's links survive, but its
    level and units are gone.
    `distorts`: the distribution changes; analysis needs to account for it.
    """

    PRESERVES_RELATIONSHIPS = "preserves_relationships"
    PRESERVES_DISTRIBUTION = "preserves_distribution"
    PRESERVES_SHAPE = "preserves_shape"
    DISTORTS = "distorts"


@dataclass(frozen=True, slots=True)
class Technique:
    """One way of transforming a column, with what it needs and what it costs.

    `replaces_identifiers` marks techniques that, applied to a direct identifier,
    leave none of the original value in the output.
    """

    name: str
    family: TechniqueFamily
    effect: StatisticalEffect
    kinds: frozenset[ColumnKind]
    function: TechniqueFunction
    summary: str
    inverse: TechniqueFunction | None = None
    linkable: bool = False
    unlinking_flag: str | None = None
    replaces_identifiers: bool = False

    @property
    def reversible(self) -> bool:
        """True when the secret key's holder can recover the original values."""
        return self.inverse is not None

    def keeps_link(self, params: Mapping[str, Any]) -> bool:
        """True when the output can still be tied back to the person.

        Reversible techniques always keep the link. Linkable techniques give the same
        output for the same person, or let anyone test a guessed value against the
        output, unless their boolean `unlinking_flag` parameter is false (its default
        applies when the plan leaves it out).
        """
        if self.reversible:
            return True
        if not self.linkable:
            return False
        if self.unlinking_flag is None:
            return True
        default = inspect.signature(self.function).parameters[self.unlinking_flag].default
        return bool(params.get(self.unlinking_flag, default))

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
        signature.bind(None, None, None, **params)
        if self.inverse is not None:
            inspect.signature(self.inverse).bind(None, None, None, **params)

    def apply(
        self, table: Table, column: str, context: TechniqueContext, params: Mapping[str, Any]
    ) -> None:
        """Replace one column of the table with its transformed values."""
        table.replace(column, self.function(table, column, context, **params))

    def reverse(
        self, table: Table, column: str, context: TechniqueContext, params: Mapping[str, Any]
    ) -> None:
        """Replace a transformed column with its recovered original values."""
        if self.inverse is None:
            raise ValueError(f"{self.name} cannot be reversed")
        table.replace(column, self.inverse(table, column, context, **params))


def apply_to_series(
    technique: Technique,
    values: pd.Series,
    context: TechniqueContext,
    params: Mapping[str, Any],
) -> pd.Series:
    """Run a technique on a single pandas column and return the new column."""
    name = "value" if values.name is None else str(values.name)
    table = Table.from_native(values.rename(name).to_frame())
    technique.apply(table, name, context, params)
    return table.to_native()[name].rename(values.name)


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

_RESCALE = TechniqueFamily.RESCALING

_RELATIONSHIPS = StatisticalEffect.PRESERVES_RELATIONSHIPS
_SHAPE = StatisticalEffect.PRESERVES_SHAPE
_DISTRIBUTION = StatisticalEffect.PRESERVES_DISTRIBUTION
_DISTORTS = StatisticalEffect.DISTORTS

_BUILTINS = (
    Technique("keep", _KEEP, _RELATIONSHIPS, _ALL_KINDS, masking.keep, "Leave unchanged."),
    Technique(
        "redact",
        _MASKING,
        _DISTORTS,
        _ALL_KINDS,
        masking.redact,
        "Replace with a constant.",
        replaces_identifiers=True,
    ),
    Technique(
        "suppress",
        _MASKING,
        _DISTORTS,
        _ALL_KINDS,
        masking.suppress,
        "Replace with missing.",
        replaces_identifiers=True,
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
        replaces_identifiers=True,
    ),
    Technique(
        "scramble_characters",
        _PSEUDO,
        _RELATIONSHIPS,
        _TEXT_LIKE,
        pseudo.scramble_characters,
        "Replace digits with digits and letters with letters, keeping the format.",
        linkable=True,
        replaces_identifiers=True,
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
        replaces_identifiers=True,
    ),
    Technique(
        "argon2_hash",
        _PSEUDO,
        _RELATIONSHIPS,
        _ALL_KINDS,
        pseudo.argon2_hash,
        "Replace with a slow, memory-hard Argon2id hash; a keyed salt keeps joins.",
        linkable=True,
        replaces_identifiers=True,
    ),
    Technique(
        "bcrypt_hash",
        _PSEUDO,
        _RELATIONSHIPS,
        _ALL_KINDS,
        pseudo.bcrypt_hash,
        "Replace with a slow bcrypt hash; a keyed salt keeps joins.",
        linkable=True,
        replaces_identifiers=True,
    ),
    Technique(
        "encrypt",
        _ENCRYPT,
        _RELATIONSHIPS,
        _ALL_KINDS,
        encryption.encrypt,
        "Reversible AES-SIV token (deterministic) or AES-GCM token (randomized).",
        inverse=encryption.decrypt,
        replaces_identifiers=True,
    ),
    Technique(
        "format_preserving_encrypt",
        _ENCRYPT,
        _RELATIONSHIPS,
        _TEXT_LIKE,
        encryption.format_preserving_encrypt,
        "Reversible NIST FF1 encryption that keeps length, format and separators.",
        inverse=encryption.format_preserving_decrypt,
        replaces_identifiers=True,
    ),
    Technique(
        "scramble_categories",
        _PSEUDO,
        _RELATIONSHIPS,
        _TEXT_LIKE | _CATEGORICAL,
        pseudo.scramble_categories,
        "Replace labels with opaque codes in random order; rare labels merged.",
        linkable=True,
        unlinking_flag="keyed",
        replaces_identifiers=True,
    ),
    Technique(
        "normalize",
        _RESCALE,
        _SHAPE,
        _NUMERIC,
        rescaling.normalize,
        "Recentre and rescale to hide level and units; shape kept unless allowed.",
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
