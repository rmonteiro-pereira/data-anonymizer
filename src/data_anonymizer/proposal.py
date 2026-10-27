"""Turn detector findings into a proposed plan, following the chosen protection and fidelity.

Direct identifiers follow one function, `_direct_identifier_rule`, because their
treatment depends only on the protection and fidelity. Every other category is a
row in `_ATTRIBUTE_RULES`, with `_ANONYMIZATION_RULES` overriding the rows that
must change when no link to people may remain.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from data_anonymizer.detection import ColumnFinding
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from data_anonymizer.surrogates import SURROGATE_CATEGORIES
from data_anonymizer.taxonomy import DataCategory, Sensitivity, sensitivity_of
from data_anonymizer.techniques import BUILTIN_TECHNIQUES, Technique

# Findings below this confidence are still proposed, but the reason asks for review.
_REVIEW_CONFIDENCE = 0.6

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
    protection: Protection = Protection.PSEUDONYMIZATION,
    techniques: Mapping[str, Technique] = BUILTIN_TECHNIQUES,
) -> AnonymizationPlan:
    """Return a plan with one action for every column that needs treatment."""
    actions = []
    for finding in findings:
        action = _action_for(finding, protection, fidelity, techniques)
        if action is not None:
            actions.append(action)

    enforce_k = protection is Protection.ANONYMIZATION or fidelity is Fidelity.PRIVACY
    return AnonymizationPlan(
        fidelity=fidelity,
        protection=protection,
        actions=actions,
        direct_identifiers=_columns_with(findings, Sensitivity.DIRECT_IDENTIFIER),
        quasi_identifiers=_columns_with(findings, Sensitivity.QUASI_IDENTIFIER),
        sensitive_attributes=_columns_with(findings, Sensitivity.SENSITIVE),
        enforce_k_anonymity=enforce_k,
    )


def _action_for(
    finding: ColumnFinding,
    protection: Protection,
    fidelity: Fidelity,
    techniques: Mapping[str, Technique],
) -> ColumnAction | None:
    rule = rule_for(finding.category, protection, fidelity)
    if rule is None:
        return None

    technique = techniques[rule.technique]
    if finding.kind not in technique.kinds:
        rule = _fallback(finding, protection)

    reason = f"{finding.category.value} ({finding.sensitivity.value})"
    if rule.note:
        reason += f": {rule.note}"
    if finding.confidence < _REVIEW_CONFIDENCE:
        reason += f"; low confidence {finding.confidence:.0%}, please review"
    return ColumnAction(
        column=finding.column,
        technique=rule.technique,
        params=dict(rule.params),
        reason=reason,
    )


def _fallback(finding: ColumnFinding, protection: Protection) -> Rule:
    # The usual technique cannot handle this storage kind, such as dates kept as
    # unparseable text. Identifiers still get removed; anything else is kept for review.
    stored_as = finding.kind.value
    if protection is Protection.ANONYMIZATION:
        return Rule("suppress", note=f"stored as {stored_as}, so removed")
    if finding.sensitivity is Sensitivity.DIRECT_IDENTIFIER:
        return Rule("keyed_hash", note=f"stored as {stored_as}, so hashed")
    return Rule("keep", note=f"stored as {stored_as}; choose a technique by hand")


def _columns_with(findings: Sequence[ColumnFinding], sensitivity: Sensitivity) -> list[str]:
    return [finding.column for finding in findings if finding.sensitivity is sensitivity]
