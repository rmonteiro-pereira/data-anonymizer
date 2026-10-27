"""Turn detector findings into a proposed plan, following the chosen fidelity.

The rules are a table, one row per data category, so changing what the library
proposes for, say, ages means editing one row.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from data_anonymizer.detection import ColumnFinding
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Fidelity
from data_anonymizer.taxonomy import DataCategory, Sensitivity
from data_anonymizer.techniques import BUILTIN_TECHNIQUES, Technique

# Findings below this confidence are still proposed, but the reason asks for review.
_REVIEW_CONFIDENCE = 0.6


@dataclass(frozen=True, slots=True)
class Rule:
    """The technique and parameters proposed for a category under one fidelity."""

    technique: str
    params: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""


def _surrogate(category: DataCategory) -> Rule:
    return Rule("surrogate", {"category": category.value}, "realistic fake, consistent per value")


_REDACT = Rule("redact", note="direct identifier removed")

_RULES: dict[DataCategory, dict[Fidelity, Rule]] = {
    DataCategory.EMAIL: {Fidelity.STATISTICAL: _surrogate(DataCategory.EMAIL)},
    DataCategory.PHONE: {Fidelity.STATISTICAL: _surrogate(DataCategory.PHONE)},
    DataCategory.PERSON_NAME: {Fidelity.STATISTICAL: _surrogate(DataCategory.PERSON_NAME)},
    DataCategory.CPF: {Fidelity.STATISTICAL: _surrogate(DataCategory.CPF)},
    DataCategory.CNPJ: {Fidelity.STATISTICAL: _surrogate(DataCategory.CNPJ)},
    DataCategory.SSN: {Fidelity.STATISTICAL: _surrogate(DataCategory.SSN)},
    DataCategory.CREDIT_CARD: {Fidelity.STATISTICAL: _surrogate(DataCategory.CREDIT_CARD)},
    DataCategory.IBAN: {Fidelity.STATISTICAL: _surrogate(DataCategory.IBAN)},
    DataCategory.IP_ADDRESS: {Fidelity.STATISTICAL: _surrogate(DataCategory.IP_ADDRESS)},
    DataCategory.STREET_ADDRESS: {
        Fidelity.STATISTICAL: _surrogate(DataCategory.STREET_ADDRESS),
    },
    DataCategory.URL: {Fidelity.STATISTICAL: _surrogate(DataCategory.URL)},
    DataCategory.RECORD_ID: {
        Fidelity.STATISTICAL: Rule("keyed_hash", note="joins and distinct counts survive"),
        Fidelity.PRIVACY: Rule(
            "suppress", note="use keyed_hash instead if rows must stay linkable"
        ),
    },
    DataCategory.FREE_TEXT_WITH_PII: {
        Fidelity.STATISTICAL: Rule("redact_entities", note="personal data inside text labelled"),
        Fidelity.PRIVACY: Rule("redact_entities", note="personal data inside text labelled"),
    },
    DataCategory.POSTAL_CODE: {
        Fidelity.STATISTICAL: Rule("truncate_string", {"keep": 3}, "region kept"),
        Fidelity.PRIVACY: Rule("truncate_string", {"keep": 2}, "broad region kept"),
    },
    DataCategory.CITY: {
        Fidelity.STATISTICAL: Rule("keep", note="kept for analysis; counted as quasi-identifier"),
        Fidelity.PRIVACY: Rule(
            "suppress_rare", {"min_count": 10, "replacement": "OTHER"}, "small towns merged"
        ),
    },
    DataCategory.DATE_OF_BIRTH: {
        Fidelity.STATISTICAL: Rule("date_shift", {"max_days": 30}, "ages stay within a month"),
        Fidelity.PRIVACY: Rule("truncate_date", {"unit": "year"}, "birth year only"),
    },
    DataCategory.AGE: {
        Fidelity.STATISTICAL: Rule("microaggregate", {"group_size": 5}, "exact mean kept"),
        Fidelity.PRIVACY: Rule("bin_numeric", {"width": 10}, "ages in decades"),
    },
    DataCategory.GENDER: {
        Fidelity.STATISTICAL: Rule("keep", note="kept; counted as quasi-identifier"),
        Fidelity.PRIVACY: Rule("keep", note="kept; counted as quasi-identifier"),
    },
    DataCategory.GEO_COORDINATE: {
        Fidelity.STATISTICAL: Rule(
            "gaussian_noise",
            {"scale": 0.01, "preserve_variance": False},
            "about one kilometre of noise",
        ),
        Fidelity.PRIVACY: Rule("round_numeric", {"decimals": 1}, "about ten kilometres"),
    },
    DataCategory.INCOME: {
        Fidelity.STATISTICAL: Rule(
            "rank_swap", {"max_distance_percent": 5.0}, "exact distribution kept"
        ),
        Fidelity.PRIVACY: Rule(
            "bin_numeric", {"bins": 5, "strategy": "quantile"}, "income quintiles"
        ),
    },
    DataCategory.HEALTH: {
        Fidelity.STATISTICAL: Rule("keep", note="kept; counted as sensitive attribute"),
        Fidelity.PRIVACY: Rule(
            "randomized_response", {"keep_probability": 0.75}, "each value deniable"
        ),
    },
}


def rule_for(category: DataCategory, fidelity: Fidelity) -> Rule | None:
    """Return the proposed rule for a category, or None when nothing is proposed."""
    rules = _RULES.get(category)
    if rules is None:
        return None
    if fidelity in rules:
        return rules[fidelity]
    # Categories listed with only a statistical rule are direct identifiers,
    # which privacy fidelity always removes.
    return _REDACT


def propose_plan(
    findings: Sequence[ColumnFinding],
    *,
    fidelity: Fidelity = Fidelity.STATISTICAL,
    techniques: Mapping[str, Technique] = BUILTIN_TECHNIQUES,
) -> AnonymizationPlan:
    """Return a plan with one action for every column that needs treatment."""
    actions = []
    for finding in findings:
        action = _action_for(finding, fidelity, techniques)
        if action is not None:
            actions.append(action)

    return AnonymizationPlan(
        fidelity=fidelity,
        actions=actions,
        quasi_identifiers=_columns_with(findings, Sensitivity.QUASI_IDENTIFIER),
        sensitive_attributes=_columns_with(findings, Sensitivity.SENSITIVE),
        enforce_k_anonymity=fidelity is Fidelity.PRIVACY,
    )


def _action_for(
    finding: ColumnFinding, fidelity: Fidelity, techniques: Mapping[str, Technique]
) -> ColumnAction | None:
    rule = rule_for(finding.category, fidelity)
    if rule is None:
        return None

    technique = techniques[rule.technique]
    if finding.kind not in technique.kinds:
        rule = _fallback(finding)

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


def _fallback(finding: ColumnFinding) -> Rule:
    # The usual technique cannot handle this storage kind, such as dates kept as
    # unparseable text. Identifiers still get removed; anything else is kept for review.
    if finding.sensitivity is Sensitivity.DIRECT_IDENTIFIER:
        return Rule("keyed_hash", note=f"stored as {finding.kind.value}, so hashed")
    return Rule("keep", note=f"stored as {finding.kind.value}; choose a technique by hand")


def _columns_with(findings: Sequence[ColumnFinding], sensitivity: Sensitivity) -> list[str]:
    return [finding.column for finding in findings if finding.sensitivity is sensitivity]
