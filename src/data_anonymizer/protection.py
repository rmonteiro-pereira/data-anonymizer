"""Decide whether an output is anonymized, pseudonymized or still identifiable.

The checks follow the three criteria of the EDPB's 2026 anonymisation guidelines:
- no record isolation: every quasi-identifier group holds at least k rows, and no
  row is unique across its discrete columns other than the sensitive attributes,
  whose spread l-diversity measures;
- no linkage: no keyed, consistent or reversible technique, and noisy numbers do
  not lead back to their records by nearest value;
- no inference: each group holds at least l distinct sensitive values.
They are necessary, not sufficient. Whether outside data an attacker holds allows
linkage, and who will receive the output, can only be judged in context; the
caveats say so.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from data_anonymizer.evaluation import PrivacyReport
from data_anonymizer.evaluation.disclosure import linkage_rate
from data_anonymizer.plan import AnonymizationPlan, Protection
from data_anonymizer.techniques import Technique, TechniqueFamily

# A column with at most this many distinct values counts as discrete for the
# whole-record uniqueness check; continuous columns make every row unique trivially.
_DISCRETE_LIMIT = 50

# A noisy column fails the linkage check when nearest-value matching re-links more
# than one record in k_target: worse odds than k-anonymity already accepts.
_LINKAGE_TOLERANCE = 1e-9

_PERTURBING_FAMILIES = {TechniqueFamily.NOISE, TechniqueFamily.PERTURBATION}

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
    untreated = _untreated_identifiers(plan)
    links = linking_actions(plan, techniques)
    exposures = (
        _exposures(plan, privacy)
        + _isolated_records(plan, released)
        + _relinkable_columns(plan, techniques, original, released)
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
    return ProtectionAssessment(
        level=level,
        target=plan.protection,
        reached=reached,
        problems=untreated + links + exposures,
        caveats=caveats,
    )


def _untreated_identifiers(plan: AnonymizationPlan) -> list[str]:
    problems = []
    for column in plan.direct_identifiers:
        action = plan.action_for(column)
        if action is None or action.technique == "keep":
            problems.append(f"{column} is a direct identifier left as it was")
    return problems


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
