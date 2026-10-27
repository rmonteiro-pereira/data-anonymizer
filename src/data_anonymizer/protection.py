"""Decide whether an output is anonymized, pseudonymized or still identifiable.

The checks follow the three criteria of the EDPB's 2026 anonymisation guidelines:
- no record isolation: every quasi-identifier group holds at least k rows, and no
  row is unique across its discrete columns other than the sensitive attributes,
  whose spread l-diversity measures;
- no linkage: no keyed, consistent or reversible technique, and noisy numbers do
  not lead back to their records by nearest value;
- no inference: each group holds at least l distinct sensitive values, and no
  sensitive number left in its original units lets one contributor's size be read
  off a segment it dominates.
They are necessary, not sufficient. Whether outside data an attacker holds allows
linkage, and who will receive the output, can only be judged in context; the
caveats say so.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from data_anonymizer.detection.patterns import TEXT_RECOGNIZERS
from data_anonymizer.evaluation import PrivacyReport
from data_anonymizer.evaluation.disclosure import linkage_rate
from data_anonymizer.evaluation.dominance import dominant_segments, keeps_original_units
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Protection
from data_anonymizer.techniques import Technique, TechniqueFamily

# A column with at most this many distinct values counts as discrete for the
# whole-record uniqueness check; continuous columns make every row unique trivially.
_DISCRETE_LIMIT = 50

# A noisy column fails the linkage check when nearest-value matching re-links more
# than one record in k_target: worse odds than k-anonymity already accepts.
_LINKAGE_TOLERANCE = 1e-9

_PERTURBING_FAMILIES = {TechniqueFamily.NOISE, TechniqueFamily.PERTURBATION}

# Share of a direct identifier's cells that may coincide with the original by chance,
# such as a random fake email that happens to exist.
_SURVIVING_LIMIT = 0.01

# An untreated column with at least this share of distinct values singles out rows.
_FINGERPRINT_SHARE = 0.9
_FINGERPRINT_MIN_ROWS = 20

_CAVEATS = (
    "Columns the detector did not flag were not treated; review the scan.",
    "Free text can hold names and other details the detector does not recognize.",
    "Only the listed quasi-identifiers were counted; others, or outside data an "
    "attacker holds, can still single people out.",
    "These automated checks passed; a contextual assessment of who receives the data "
    "and what they could combine it with is still required.",
)


class ProtectionLevel(StrEnum):
    """What an output turned out to be.

    `anonymized`: no link to people and every check passed. `at_risk`: no link to
    people, but some records can still be singled out or their sensitive values
    inferred. `pseudonymized`: identifiers replaced, but the key holder can re-link
    them. `identifiable`: a direct identifier was left as it was.
    """

    ANONYMIZED = "anonymized"
    AT_RISK = "at_risk"
    PSEUDONYMIZED = "pseudonymized"
    IDENTIFIABLE = "identifiable"


@dataclass(frozen=True, slots=True)
class ProtectionAssessment:
    """The level reached, whether it meets the plan's target, and why not."""

    level: ProtectionLevel
    target: Protection
    reached: bool
    problems: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def linking_actions(plan: AnonymizationPlan, techniques: Mapping[str, Technique]) -> list[str]:
    """Return a description of every action whose output can be tied back to a person."""
    return [
        f"{action.column} uses {action.technique}, which keeps a link to the person"
        for action in plan.actions
        if techniques[action.technique].keeps_link(action.params)
    ]


def assess_protection(
    plan: AnonymizationPlan,
    techniques: Mapping[str, Technique],
    privacy: PrivacyReport,
    original: pd.DataFrame,
    released: pd.DataFrame,
) -> ProtectionAssessment:
    """Return the protection level an applied plan reached."""
    untreated = _untreated_identifiers(plan, techniques, original, released)
    links = linking_actions(plan, techniques)
    exposures = (
        _exposures(plan, privacy)
        + _isolated_records(plan, released)
        + _fingerprint_columns(plan, released)
        + _relinkable_columns(plan, techniques, original, released)
        + _magnitude_disclosures(plan, original, released)
    )

    if untreated:
        level = ProtectionLevel.IDENTIFIABLE
    elif links:
        level = ProtectionLevel.PSEUDONYMIZED
    elif exposures:
        level = ProtectionLevel.AT_RISK
    else:
        level = ProtectionLevel.ANONYMIZED

    if plan.protection is Protection.ANONYMIZATION:
        reached = level is ProtectionLevel.ANONYMIZED
    else:
        reached = level is not ProtectionLevel.IDENTIFIABLE
    caveats = list(_CAVEATS) if level is ProtectionLevel.ANONYMIZED else []
    caveats += _rescaling_caveats(plan)
    return ProtectionAssessment(
        level=level,
        target=plan.protection,
        reached=reached,
        problems=untreated + links + exposures,
        caveats=caveats,
    )


def _untreated_identifiers(
    plan: AnonymizationPlan,
    techniques: Mapping[str, Technique],
    original: pd.DataFrame,
    released: pd.DataFrame,
) -> list[str]:
    problems = []
    for column in plan.direct_identifiers:
        action = plan.action_for(column)
        if action is None or not _replaces(action, techniques):
            problems.append(f"{column} is a direct identifier its technique does not replace")
            continue
        if action.technique == "redact_entities":
            remaining = _share_still_holding_entities(released[column])
            if remaining > _SURVIVING_LIMIT:
                problems.append(
                    f"{column}: {remaining:.0%} of texts still hold recognizable personal data"
                )
            continue
        surviving = _surviving_share(original[column], released[column])
        if surviving > _SURVIVING_LIMIT:
            problems.append(f"{column}: {surviving:.0%} of identifier values survive unchanged")
    return problems


def _share_still_holding_entities(texts: pd.Series) -> float:
    present = texts.dropna().astype(str)
    if present.empty:
        return 0.0
    holding = present.map(
        lambda text: any(recognizer.find_spans(text) for recognizer in TEXT_RECOGNIZERS)
    )
    return float(holding.mean())


def _replaces(action: ColumnAction, techniques: Mapping[str, Technique]) -> bool:
    if action.technique == "redact_entities":
        # Free text is cleaned by labelling what it contains; the check on the
        # released text catches a run limited to some categories.
        return True
    return techniques[action.technique].replaces_identifiers


def _surviving_share(before: pd.Series, after: pd.Series) -> float:
    present = before.notna() & after.notna()
    if not present.any():
        return 0.0
    same = before[present].astype(str) == after[present].astype(str)
    return float(same.mean())


def _exposures(plan: AnonymizationPlan, privacy: PrivacyReport) -> list[str]:
    problems = []
    if privacy.k_anonymity < plan.k_target:
        problems.append(
            f"{privacy.rows_below_target} rows sit in quasi-identifier groups smaller than "
            f"{plan.k_target} (k-anonymity is {privacy.k_anonymity})"
        )
    for attribute, diversity in privacy.l_diversity.items():
        if diversity < plan.l_target:
            problems.append(
                f"some groups hold only {diversity} distinct value(s) of {attribute}, "
                f"below the l-diversity target of {plan.l_target}"
            )
    return problems


def _fingerprint_columns(plan: AnonymizationPlan, released: pd.DataFrame) -> list[str]:
    # A column released as it was, with a different value in nearly every row, is a
    # fingerprint: anyone holding that value elsewhere finds the row.
    treated = {action.column for action in plan.actions if action.technique != "keep"}
    removed = set(plan.direct_identifiers)
    problems = []
    for column in released.columns:
        if column in treated or column in removed:
            continue
        present = released[column].dropna()
        if len(present) < _FINGERPRINT_MIN_ROWS:
            continue
        unique_share = present.astype(str).nunique() / len(present)
        if unique_share >= _FINGERPRINT_SHARE:
            problems.append(
                f"{column} is released unchanged and {unique_share:.0%} of its values are "
                "distinct, so it can single out rows"
            )
    return problems


def _magnitude_disclosures(
    plan: AnonymizationPlan, original: pd.DataFrame, released: pd.DataFrame
) -> list[str]:
    contributors = _contributors(plan, original)
    groupings = [
        column
        for column in plan.quasi_identifiers
        if released[column].nunique(dropna=False) <= _DISCRETE_LIMIT
    ]
    problems = []
    for attribute in plan.sensitive_attributes:
        if not pd.api.types.is_numeric_dtype(original[attribute]):
            continue
        if not keeps_original_units(original[attribute], released[attribute]):
            continue
        for grouping in groupings:
            for segment in dominant_segments(
                original[attribute],
                released[grouping],
                contributors,
                group_column=grouping,
            ):
                problems.append(
                    f"{attribute} keeps its original units, and in {grouping} = "
                    f"{segment.group!r} one contributor holds {segment.largest_share:.0%} "
                    f"of the total ({segment.rule}); normalize it or merge segments"
                )
    return problems


def _contributors(plan: AnonymizationPlan, original: pd.DataFrame) -> pd.Series:
    if plan.entity_column is not None:
        return original[plan.entity_column]
    return pd.Series(original.index, index=original.index)


def _rescaling_caveats(plan: AnonymizationPlan) -> list[str]:
    affine = [
        action.column
        for action in plan.actions
        if action.technique == "normalize" and not action.params.get("allow_shape_change")
    ]
    if not affine:
        return []
    return [
        f"{', '.join(affine)} were shifted and scaled: anyone who knows the true value of "
        "two records can undo the scale. Group by entity with allow_shape_change to "
        "remove that risk."
    ]


def _isolated_records(plan: AnonymizationPlan, released: pd.DataFrame) -> list[str]:
    removed = set(plan.direct_identifiers) | set(plan.sensitive_attributes)
    discrete = [
        column
        for column in released.columns
        if column not in removed and released[column].nunique(dropna=False) <= _DISCRETE_LIMIT
    ]
    columns = list(dict.fromkeys([*plan.quasi_identifiers, *discrete]))
    columns = [column for column in columns if column in released.columns]
    if not columns or released.empty:
        return []
    keys = released[columns].astype(str).assign(_row=1)
    sizes = keys.groupby(columns, dropna=False)["_row"].transform("size")
    unique_rows = int((sizes == 1).sum())
    if unique_rows == 0:
        return []
    return [
        f"{unique_rows} rows are unique across their discrete columns "
        f"({', '.join(columns)}), so they can be singled out"
    ]


def _relinkable_columns(
    plan: AnonymizationPlan,
    techniques: Mapping[str, Technique],
    original: pd.DataFrame,
    released: pd.DataFrame,
) -> list[str]:
    problems = []
    for action in plan.actions:
        if techniques[action.technique].family not in _PERTURBING_FAMILIES:
            continue
        before = pd.to_numeric(original[action.column], errors="coerce")
        after = pd.to_numeric(released[action.column], errors="coerce")
        both = before.notna() & after.notna()
        if both.sum() < 2:
            continue
        rate = linkage_rate(before[both].to_numpy(), after[both].to_numpy())
        limit = 1 / plan.k_target
        if rate > limit + _LINKAGE_TOLERANCE:
            problems.append(
                f"{action.column}: {rate:.0%} of noisy values lead back to their record by "
                f"nearest value, more than 1 in {plan.k_target}; add noise or generalize"
            )
    return problems
