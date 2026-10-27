"""Turn detector findings into a proposed plan, following the chosen protection and fidelity.

Direct identifiers follow one function, `_direct_identifier_rule`, because their
treatment depends only on the protection and fidelity. Every other category is a
row in `_ATTRIBUTE_RULES`, with `_ANONYMIZATION_RULES` overriding the rows that
must change when no link to people may remain.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw

from data_anonymizer.detection import ColumnFinding
from data_anonymizer.engines import WorkingFrame
from data_anonymizer.identifying_columns import untreated_discrete_columns
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from data_anonymizer.surrogates import SURROGATE_CATEGORIES
from data_anonymizer.taxonomy import ColumnKind, DataCategory, Sensitivity, sensitivity_of
from data_anonymizer.techniques import BUILTIN_TECHNIQUES, Technique

# Findings below this confidence are still proposed, but the reason asks for review.
_REVIEW_CONFIDENCE = 0.6

# Techniques whose random draw is taken once per person when the plan names an entity column.
_PER_ENTITY_DRAW = ("date_shift", "randomized_response")

_STATISTICAL = Fidelity.STATISTICAL
_PRIVACY = Fidelity.PRIVACY


@dataclass(frozen=True, slots=True)
class Rule:
    """The technique and parameters proposed for one category."""

    technique: str
    params: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""


_ATTRIBUTE_RULES: dict[DataCategory, dict[Fidelity, Rule]] = {
    DataCategory.URL: {
        _STATISTICAL: Rule(
            "surrogate", {"category": "url"}, "realistic fake, consistent per value"
        ),
        _PRIVACY: Rule("keyed_hash", note="linkable only with the secret key"),
    },
    DataCategory.POSTAL_CODE: {
        _STATISTICAL: Rule("truncate_string", {"keep": 3}, "region kept"),
        _PRIVACY: Rule("truncate_string", {"keep": 2}, "broad region kept"),
    },
    DataCategory.CITY: {
        _STATISTICAL: Rule("keep", note="kept for analysis; counted as quasi-identifier"),
        _PRIVACY: Rule(
            "suppress_rare", {"min_count": 10, "replacement": "OTHER"}, "small towns merged"
        ),
    },
    DataCategory.DATE_OF_BIRTH: {
        _STATISTICAL: Rule("date_shift", {"max_days": 30}, "ages stay within a month"),
        _PRIVACY: Rule("truncate_date", {"unit": "year"}, "birth year only"),
    },
    DataCategory.AGE: {
        _STATISTICAL: Rule("microaggregate", {"group_size": 5}, "exact mean kept"),
        _PRIVACY: Rule("bin_numeric", {"width": 10}, "ages in decades"),
    },
    DataCategory.GENDER: {
        _STATISTICAL: Rule("keep", note="kept; counted as quasi-identifier"),
        _PRIVACY: Rule("keep", note="kept; counted as quasi-identifier"),
    },
    DataCategory.GEO_COORDINATE: {
        _STATISTICAL: Rule(
            "gaussian_noise",
            {"scale": 0.01, "preserve_variance": False},
            "about one kilometre of noise",
        ),
        _PRIVACY: Rule("round_numeric", {"decimals": 1}, "about ten kilometres"),
    },
    DataCategory.INCOME: {
        _STATISTICAL: Rule("rank_swap", {"max_distance_percent": 5.0}, "exact distribution kept"),
        _PRIVACY: Rule("bin_numeric", {"bins": 5, "strategy": "quantile"}, "income quintiles"),
    },
    DataCategory.HEALTH: {
        _STATISTICAL: Rule("keep", note="kept; counted as sensitive attribute"),
        _PRIVACY: Rule("randomized_response", {"keep_probability": 0.75}, "each value deniable"),
    },
    DataCategory.SPECIAL_CATEGORY: {
        _STATISTICAL: Rule("keep", note="kept; counted as sensitive attribute"),
        _PRIVACY: Rule("randomized_response", {"keep_probability": 0.75}, "each value deniable"),
    },
    DataCategory.NATIONALITY: {
        _STATISTICAL: Rule(
            "suppress_rare", {"min_count": 5, "replacement": "OTHER"}, "rare countries merged"
        ),
        _PRIVACY: Rule(
            "suppress_rare", {"min_count": 20, "replacement": "OTHER"}, "rare countries merged"
        ),
    },
    DataCategory.MARITAL_STATUS: {
        _STATISTICAL: Rule("keep", note="kept; counted as quasi-identifier"),
        _PRIVACY: Rule("keep", note="kept; counted as quasi-identifier"),
    },
    DataCategory.BUSINESS_SEGMENT: {
        _STATISTICAL: Rule(
            "scramble_categories",
            {"prefix": "SEG", "min_count": 3, "min_entities": 3},
            "opaque codes; segments with fewer than 3 rows or 3 entities merged",
        ),
        _PRIVACY: Rule(
            "scramble_categories",
            {"prefix": "SEG", "min_count": 5, "min_entities": 5},
            "opaque codes; segments with fewer than 5 rows or 5 entities merged",
        ),
    },
    DataCategory.FLOOR_AREA: {
        _STATISTICAL: Rule(
            "bin_numeric", {"bins": 4, "strategy": "quantile"}, "size quartiles; anchors hidden"
        ),
        _PRIVACY: Rule(
            "bin_numeric", {"bins": 4, "strategy": "quantile"}, "size quartiles; anchors hidden"
        ),
    },
    DataCategory.BUSINESS_REVENUE: {
        _STATISTICAL: Rule("normalize", {"method": "robust"}, "units removed; shape kept"),
        _PRIVACY: Rule("normalize", {"method": "robust"}, "units removed; shape kept"),
    },
    DataCategory.RENT: {
        _STATISTICAL: Rule("normalize", {"method": "robust"}, "units removed; shape kept"),
        _PRIVACY: Rule("normalize", {"method": "robust"}, "units removed; shape kept"),
    },
}

# Under anonymization, values that stay exact or stay unique per person are replaced
# by values shared by several people.
_ANONYMIZATION_RULES: dict[DataCategory, dict[Fidelity, Rule]] = {
    DataCategory.URL: {
        _STATISTICAL: Rule(
            "surrogate", {"category": "url", "consistent": False}, "random fake per cell"
        ),
        _PRIVACY: Rule("suppress", note="removed"),
    },
    DataCategory.DATE_OF_BIRTH: {
        _STATISTICAL: Rule("truncate_date", {"unit": "year"}, "birth year only"),
        _PRIVACY: Rule("truncate_date", {"unit": "year"}, "birth year only"),
    },
    DataCategory.GEO_COORDINATE: {
        _STATISTICAL: Rule("round_numeric", {"decimals": 2}, "about one kilometre"),
        _PRIVACY: Rule("round_numeric", {"decimals": 1}, "about ten kilometres"),
    },
    DataCategory.INCOME: {
        _STATISTICAL: Rule(
            "microaggregate", {"group_size": 5}, "each value shared by five people; mean kept"
        ),
        _PRIVACY: Rule("bin_numeric", {"bins": 5, "strategy": "quantile"}, "income quintiles"),
    },
}


def rule_for(category: DataCategory, protection: Protection, fidelity: Fidelity) -> Rule | None:
    """Return the proposed rule for a category, or None when nothing is proposed."""
    if sensitivity_of(category) is Sensitivity.DIRECT_IDENTIFIER:
        return _direct_identifier_rule(category, protection, fidelity)
    if protection is Protection.ANONYMIZATION and category in _ANONYMIZATION_RULES:
        return _ANONYMIZATION_RULES[category][fidelity]
    rules = _ATTRIBUTE_RULES.get(category)
    if rules is None:
        return None
    return rules[fidelity]


def _direct_identifier_rule(
    category: DataCategory, protection: Protection, fidelity: Fidelity
) -> Rule:
    if category is DataCategory.FREE_TEXT_WITH_PII:
        return Rule("redact_entities", note="personal data inside text labelled")
    if category is DataCategory.ORGANIZATION:
        # Codes keep each company's rows together, so its behaviour can still be modelled.
        return Rule("scramble_categories", {"prefix": "ORG"}, "opaque code per company")
    if (
        category is DataCategory.RECORD_ID
        and protection is Protection.ANONYMIZATION
        and fidelity is _STATISTICAL
    ):
        return Rule("scramble_categories", {"prefix": "ID"}, "random code; rows stay grouped")
    has_fakes = category in SURROGATE_CATEGORIES and category is not DataCategory.RECORD_ID

    if protection is Protection.PSEUDONYMIZATION:
        if fidelity is _STATISTICAL and has_fakes:
            return Rule(
                "surrogate", {"category": category.value}, "realistic fake, consistent per value"
            )
        return Rule("keyed_hash", note="linkable only with the secret key")

    if fidelity is _STATISTICAL and has_fakes:
        return Rule(
            "surrogate",
            {"category": category.value, "consistent": False},
            "random fake per cell; no link back to the person",
        )
    return Rule("suppress", note="removed")


def propose_plan(
    findings: Sequence[ColumnFinding],
    *,
    fidelity: Fidelity = Fidelity.STATISTICAL,
    protection: Protection = Protection.ANONYMIZATION,
    techniques: Mapping[str, Technique] = BUILTIN_TECHNIQUES,
) -> AnonymizationPlan:
    """Return a plan with one action for every column that needs treatment."""
    entity_column = _entity_column(findings)
    actions = []
    for finding in findings:
        action = _action_for(finding, protection, fidelity, techniques, entity_column)
        if action is not None:
            actions.append(action)

    enforce_k = protection is Protection.ANONYMIZATION or fidelity is Fidelity.PRIVACY
    return AnonymizationPlan(
        fidelity=fidelity,
        protection=protection,
        actions=actions,
        direct_identifiers=_columns_with(findings, Sensitivity.DIRECT_IDENTIFIER),
        entity_column=entity_column,
        row_key=_record_id_column(findings),
        quasi_identifiers=_columns_with(findings, Sensitivity.QUASI_IDENTIFIER),
        sensitive_attributes=_columns_with(findings, Sensitivity.SENSITIVE),
        enforce_k_anonymity=enforce_k,
    )


def replace_untreated_keys(plan: AnonymizationPlan, frame: WorkingFrame) -> AnonymizationPlan:
    """Add a fresh-code action for every untreated discrete column of an anonymization plan.

    A column of account numbers, cards or devices ties together the rows of one entity, and
    nobody can tell it from a product code by its statistics. Fresh per-release codes keep the
    grouping for a model while nobody can look up their own value. Only text columns are
    replaced: a whole-number column may be a measurement whose distribution must survive, so it
    is left to the protection report. Pseudonymization plans are left alone, since the key
    holder may need the original values.
    """
    if plan.protection is not Protection.ANONYMIZATION:
        return plan
    schema = frame.collect_schema()
    columns = [
        column
        for column in untreated_discrete_columns(plan, frame)
        if plan.action_for(column) is None and _holds_text(schema[column])
    ]
    if not columns:
        return plan
    actions = [*plan.actions, *(_fresh_code_action(column) for column in columns)]
    return plan.model_copy(update={"actions": actions})


def _holds_text(dtype: Any) -> bool:
    return dtype == nw.String or dtype == nw.Categorical


def _fresh_code_action(column: str) -> ColumnAction:
    return ColumnAction(
        column=column,
        technique="scramble_categories",
        params={"prefix": "KEY"},
        reason="untreated discrete column: fresh code per release; rows stay grouped",
    )


def _action_for(
    finding: ColumnFinding,
    protection: Protection,
    fidelity: Fidelity,
    techniques: Mapping[str, Technique],
    entity_column: str | None,
) -> ColumnAction | None:
    rule = rule_for(finding.category, protection, fidelity)
    if rule is None:
        return None

    technique = techniques[rule.technique]
    if finding.kind not in technique.kinds:
        rule = _fallback(finding, protection, fidelity)
    params = dict(rule.params)
    if rule.technique == "scramble_categories" and protection is Protection.PSEUDONYMIZATION:
        params["keyed"] = True
    if rule.technique in _PER_ENTITY_DRAW and entity_column is not None:
        # One draw per person; separate draws per row average out across a person's rows.
        params["by"] = entity_column
    if rule.technique == "scramble_categories" and "min_entities" in params:
        _count_entities_or_drop(params, entity_column)

    reason = f"{finding.category.value} ({finding.sensitivity.value})"
    if rule.note:
        reason += f": {rule.note}"
    if finding.confidence < _REVIEW_CONFIDENCE:
        reason += f"; low confidence {finding.confidence:.0%}, please review"
    return ColumnAction(
        column=finding.column,
        technique=rule.technique,
        params=params,
        reason=reason,
    )


def _count_entities_or_drop(params: dict[str, Any], entity_column: str | None) -> None:
    # Counting entities needs a column that says whose row it is; rows are the entities otherwise.
    if entity_column is None:
        del params["min_entities"]
    else:
        params["by"] = entity_column


def _fallback(finding: ColumnFinding, protection: Protection, fidelity: Fidelity) -> Rule:
    # The usual technique cannot handle this storage kind, such as dates kept as
    # unparseable text. Identifiers still get removed; anything else is kept for review.
    stored_as = finding.kind.value
    categorical = finding.kind in (ColumnKind.CATEGORY, ColumnKind.BOOLEAN)
    if finding.sensitivity is Sensitivity.SENSITIVE and categorical:
        # Suppressing the attribute would leave nothing to analyse, such as an income
        # bracket stored as text; randomize it under privacy fidelity instead.
        if fidelity is _PRIVACY:
            return Rule(
                "randomized_response",
                {"keep_probability": 0.75},
                f"stored as {stored_as}; each value deniable",
            )
        return Rule("keep", note=f"stored as {stored_as}; counted as sensitive attribute")
    if protection is Protection.ANONYMIZATION:
        return Rule("suppress", note=f"stored as {stored_as}, so removed")
    if finding.sensitivity is Sensitivity.DIRECT_IDENTIFIER:
        return Rule("keyed_hash", note=f"stored as {stored_as}, so hashed")
    return Rule("keep", note=f"stored as {stored_as}; choose a technique by hand")


def _columns_with(findings: Sequence[ColumnFinding], sensitivity: Sensitivity) -> list[str]:
    return [finding.column for finding in findings if finding.sensitivity is sensitivity]


def _entity_column(findings: Sequence[ColumnFinding]) -> str | None:
    """The column that says whose row it is: a company, else the first record key."""
    for finding in findings:
        if finding.category is DataCategory.ORGANIZATION:
            return finding.column
    return _record_id_column(findings)


def _record_id_column(findings: Sequence[ColumnFinding]) -> str | None:
    """The first column found to hold a different value in nearly every row."""
    for finding in findings:
        if finding.category is DataCategory.RECORD_ID:
            return finding.column
    return None
