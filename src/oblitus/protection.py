"""Decide whether an output is anonymized, pseudonymized or still identifiable.

The checks follow the three criteria of the EDPB's 2026 anonymisation guidelines:
- no record isolation: every quasi-identifier group holds at least k rows, and no
  row is unique across its discrete columns other than the sensitive attributes,
  whose spread l-diversity measures;
- no linkage: no keyed, consistent or reversible technique, and noisy numbers do
  not lead back to their records by nearest value;
- no inference: each group holds at least l distinct sensitive values, no group's sensitive
  distribution lies farther than the plan's t-closeness target (when one is set) from the
  whole table's, no group lets a sensitive value be guessed with high confidence and well
  above its base rate, and no sensitive number left in its original units lets one contributor's
  size be read off a segment it dominates.
Record isolation and fingerprints are counted on the whole released table. The checks that
compare each released value with its original run on an aligned in-memory sample of at
most `MAX_SAMPLE_ROWS` rows.
They are necessary, not sufficient. Whether outside data an attacker holds allows
linkage, and who will receive the output, can only be judged in context; the
caveats say so.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import narwhals as nw
import pandas as pd

from oblitus.detection.named_entities import NamedEntityBackend, load_named_entity_backend
from oblitus.detection.patterns import TEXT_RECOGNIZERS
from oblitus.detection.text_entities import holds_entities
from oblitus.engines import WorkingFrame, first_row
from oblitus.evaluation import PrivacyReport
from oblitus.evaluation.aligned_sample import AlignedSample
from oblitus.evaluation.disclosure import joint_linkage_rate, linkage_rate
from oblitus.evaluation.dominance import dominant_segments, keeps_original_units
from oblitus.evaluation.inference import confident_inferences
from oblitus.evaluation.privacy import GROUP_SIZE, group_sizes
from oblitus.identifying_columns import entity_key_columns, fingerprint_columns
from oblitus.plan import AnonymizationPlan, ColumnAction, Protection
from oblitus.techniques import Technique, TechniqueFamily
from oblitus.techniques.rescaling import SHAPE_KEEPING_METHODS

# A column with at most this many distinct values counts as discrete for the
# whole-record uniqueness check; continuous columns make every row unique trivially.
DISCRETE_LIMIT = 50

# A noisy column fails the linkage check when nearest-value matching re-links more
# than one record in k_target: worse odds than k-anonymity already accepts.
_LINKAGE_TOLERANCE = 1e-9

_PERTURBING_FAMILIES = {TechniqueFamily.NOISE, TechniqueFamily.PERTURBATION}

# Share of a direct identifier's cells that may coincide with the original by chance,
# such as a random fake email that happens to exist.
_SURVIVING_LIMIT = 0.01

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


def with_added_exposures(
    assessment: ProtectionAssessment, exposures: list[str]
) -> ProtectionAssessment:
    """The assessment after more record-isolation or inference problems were found.

    A release that was `anonymized` becomes `at_risk`, and loses the caveats that only
    belong to an anonymized release. Other levels keep their level and gain the problems.
    """
    if not exposures:
        return assessment
    level = assessment.level
    if level is ProtectionLevel.ANONYMIZED:
        level = ProtectionLevel.AT_RISK
    if assessment.target is Protection.ANONYMIZATION:
        reached = level is ProtectionLevel.ANONYMIZED
    else:
        reached = level is not ProtectionLevel.IDENTIFIABLE
    caveats = [c for c in assessment.caveats if c not in _CAVEATS]
    return ProtectionAssessment(
        level=level,
        target=assessment.target,
        reached=reached,
        problems=[*assessment.problems, *exposures],
        caveats=caveats,
    )


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
    sample: AlignedSample,
    released_table: WorkingFrame,
) -> ProtectionAssessment:
    """Return the protection level an applied plan reached.

    `released_table` is the whole released table, for the checks that count rows.
    `sample` holds the same rows of the original and the released table, for the checks
    that compare values.
    """
    original = sample.original
    released = sample.released
    untreated = _untreated_identifiers(plan, techniques, original, released)
    links = linking_actions(plan, techniques)
    exposures = (
        _exposures(plan, privacy)
        + _isolated_records(plan, released_table)
        + _confident_inferences(plan, released_table)
        + fingerprint_columns(plan, techniques, released_table)
        + entity_key_columns(plan, released_table)
        + _free_text_leftovers(plan)
        + _affine_rescalings(plan)
        + _independent_draws(plan)
        + _relinkable_columns(plan, techniques, original, released)
        + _jointly_relinkable_columns(plan, techniques, original, released)
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
            backend = _entity_backend(action)
            remaining = _share_still_holding_entities(released[column], backend)
            if remaining > _SURVIVING_LIMIT:
                problems.append(
                    f"{column}: {remaining:.0%} of texts still hold recognizable personal data"
                )
            continue
        surviving = _surviving_share(original[column], released[column])
        if surviving > _SURVIVING_LIMIT:
            problems.append(f"{column}: {surviving:.0%} of identifier values survive unchanged")
    return problems


def _entity_backend(action: ColumnAction) -> NamedEntityBackend | None:
    entity_model = action.params.get("entity_model")
    if entity_model is None:
        return None
    return load_named_entity_backend(entity_model)


def _share_still_holding_entities(texts: pd.Series, backend: NamedEntityBackend | None) -> float:
    present = texts.dropna().astype(str)
    if present.empty:
        return 0.0
    holding = present.map(lambda text: holds_entities(text, TEXT_RECOGNIZERS, backend))
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
    if plan.t_target is not None:
        for attribute, distance in privacy.t_closeness.items():
            if distance > plan.t_target:
                problems.append(
                    f"some groups differ from the whole table in {attribute} by {distance:.2f}, "
                    f"above the t-closeness target of {plan.t_target:g}"
                )
    return problems


def _confident_inferences(plan: AnonymizationPlan, released: WorkingFrame) -> list[str]:
    if plan.protection is not Protection.ANONYMIZATION:
        return []
    findings = confident_inferences(
        released,
        plan.released_quasi_identifiers,
        plan.sensitive_attributes,
        confidence=plan.inference_confidence,
        gain=plan.inference_gain,
        discrete_limit=DISCRETE_LIMIT,
        entity_column=plan.entity_column,
    )
    return [
        f"{finding.attribute}: in {_describe_group(finding.group)} "
        f"({finding.group_rows} rows), {finding.group_share:.0%} have {finding.value!r}, "
        f"against {finding.table_share:.0%} in the whole table; {finding.exposed_groups} "
        "group(s) let an attacker guess this value with high confidence. Set t_target, "
        "generalize the quasi-identifiers further, or merge these groups"
        for finding in findings
    ]


def _describe_group(group: Mapping[str, object]) -> str:
    return ", ".join(f"{column} = {value!r}" for column, value in group.items())


def _magnitude_disclosures(
    plan: AnonymizationPlan, original: pd.DataFrame, released: pd.DataFrame
) -> list[str]:
    contributors = _contributors(plan, original)
    groupings = [
        column
        for column in plan.quasi_identifiers
        if released[column].nunique(dropna=False) <= DISCRETE_LIMIT
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


def _affinely_rescaled_columns(plan: AnonymizationPlan) -> list[str]:
    # The method and the grouping decide the arithmetic; allow_shape_change only permits it.
    return [
        action.column
        for action in plan.actions
        if action.technique == "normalize"
        and action.params.get("method", "robust") in SHAPE_KEEPING_METHODS
        and action.params.get("by") is None
    ]


def _affine_rescalings(plan: AnonymizationPlan) -> list[str]:
    # Shift and scale are two unknowns. Two true values, or one true value and the true total,
    # solve them and give back every record, so no anonymization claim survives that.
    affine = _affinely_rescaled_columns(plan)
    if not affine or plan.protection is not Protection.ANONYMIZATION:
        return []
    return [
        f"{', '.join(affine)} were only shifted and scaled: anyone who knows two true values, "
        "or one true value and the true total, can undo the scale and recover every record; "
        "use normalize grouped by the entity with allow_shape_change, or rank"
    ]


def _rescaling_caveats(plan: AnonymizationPlan) -> list[str]:
    affine = _affinely_rescaled_columns(plan)
    if not affine or plan.protection is Protection.ANONYMIZATION:
        return []
    return [
        f"{', '.join(affine)} were shifted and scaled: anyone who knows the true value of "
        "two records can undo the scale. Group by entity with allow_shape_change to "
        "remove that risk."
    ]


def _isolated_records(plan: AnonymizationPlan, released: WorkingFrame) -> list[str]:
    names = released.collect_schema().names()
    removed = set(plan.direct_identifiers) | set(plan.sensitive_attributes)
    entity = plan.entity_column if plan.entity_column in names else None
    candidates = [column for column in names if column not in removed and column != entity]
    row_count = first_row(released.select(nw.len().alias("rows")))["rows"]
    if not candidates or row_count == 0:
        return []
    distinct = first_row(released.select(*[nw.col(c).n_unique().alias(c) for c in candidates]))
    discrete = [column for column in candidates if distinct[column] <= DISCRETE_LIMIT]
    microaggregated = [
        column
        for step in plan.joint_microaggregation
        for column in step.columns
        if column in candidates
    ]
    # Microaggregated columns have many distinct values by nature, yet each record's vector over
    # them must still be shared, so they count whatever their number of distinct values.
    columns = list(dict.fromkeys([*plan.quasi_identifiers, *discrete, *microaggregated]))
    columns = [column for column in columns if column in names]
    if not columns:
        return []
    sizes = nw.col(GROUP_SIZE)
    singles = nw.when(sizes == 1).then(1).otherwise(0).sum().alias("unique_rows")
    singled_out = group_sizes(released, columns, entity).select(singles)
    unique_rows = int(first_row(singled_out)["unique_rows"])
    if unique_rows == 0:
        return []
    counted = "rows" if entity is None else "entities"
    return [
        f"{unique_rows} {counted} are unique across their discrete columns "
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


def _jointly_relinkable_columns(
    plan: AnonymizationPlan,
    techniques: Mapping[str, Technique],
    original: pd.DataFrame,
    released: pd.DataFrame,
) -> list[str]:
    """Check the perturbed numeric columns together, as one vector per record.

    Each column can pass the check on its own while the three or four of them, taken as a
    point in space, still lead to a single record. An attacker holding the originals matches
    each released vector to its nearest original vector.
    """
    columns = [
        action.column
        for action in plan.actions
        if techniques[action.technique].family in _PERTURBING_FAMILIES
        and pd.api.types.is_numeric_dtype(original[action.column])
        and pd.api.types.is_numeric_dtype(released[action.column])
    ]
    if len(columns) < 2:
        return []
    before = original[columns].apply(pd.to_numeric, errors="coerce")
    after = released[columns].apply(pd.to_numeric, errors="coerce")
    both = before.notna().all(axis=1) & after.notna().all(axis=1)
    if both.sum() < 2:
        return []
    rate = joint_linkage_rate(before[both].to_numpy(), after[both].to_numpy())
    if rate <= 1 / plan.k_target + _LINKAGE_TOLERANCE:
        return []
    return [
        f"{', '.join(columns)}: {rate:.0%} of records lead back to their original by nearest "
        f"standardized vector over these columns together, more than 1 in {plan.k_target}; "
        "add noise, generalize, or perturb fewer of them"
    ]


def _free_text_leftovers(plan: AnonymizationPlan) -> list[str]:
    # Patterns cannot find a name written into a sentence, and a model can miss one, so a
    # cleaned text column is never proven free of personal data.
    problems = []
    for action in plan.actions:
        if action.technique != "redact_entities":
            continue
        if action.params.get("entity_model") is None:
            problems.append(
                f"{action.column}: redact_entities labels emails, document numbers and phones, "
                "but names and other free-text details may remain"
            )
        else:
            problems.append(
                f"{action.column}: redact_entities also uses a model for names and addresses, "
                "but a model can miss some"
            )
    return problems


def _independent_draws(plan: AnonymizationPlan) -> list[str]:
    # One person's rows each get their own draw, and a majority vote over them recovers
    # the true value.
    if plan.entity_column is None:
        return []
    return [
        f"{action.column}: randomized_response draws again for every row of an entity, so "
        f"averaging a person's rows recovers the true value; set by={plan.entity_column!r}"
        for action in plan.actions
        if action.technique == "randomized_response" and "by" not in action.params
    ]
