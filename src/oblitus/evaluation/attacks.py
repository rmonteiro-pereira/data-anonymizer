"""Attack-based privacy risk: try to single out, link and infer, and report how often it works.

k-anonymity and the other group measures say how a table is built. These checks instead play
the attacker, with the Anonymeter library, and report what share of attacks succeed, with a
confidence interval:
- singling out: the attacker writes a query that matches exactly one original record;
- linkability: the attacker holds two sets of columns of the same people, such as columns from
  two databases, and tries to tell which released records belong together;
- inference: the attacker knows the quasi-identifiers of a person and guesses a sensitive
  attribute from the nearest released records.

Anonymeter reports a risk between 0 and 1, where 0 means the attacker did no better than a
random guess. A `control` table, original records the release was not built from, lets it
separate what the release reveals from what the population makes easy to guess. Without one,
the risk is the raw success rate of the attack.

The library is imported when a check runs, so oblitus works without it. Install the extra
with `pip install "oblitus[anonymeter]"`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from oblitus.errors import MissingExtraError

# A numeric attribute with more distinct values than this is guessed as a number.
_REGRESSION_DISTINCT = 20
_SINGLING_OUT_COLUMNS = 3
# Random queries that may be tried per attack. Columns with few distinct values rarely give a
# query that matches one record, and Anonymeter's own limit of ten million tries takes minutes.
_SINGLING_OUT_ATTEMPTS_PER_ATTACK = 200


@dataclass(frozen=True, slots=True)
class AttackRisk:
    """The success rate of one attack, with its confidence interval."""

    attack: str
    risk: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class AttackRiskReport:
    """The risk of every attack that could run, and the attacks that could not."""

    confidence_level: float
    n_attacks: int
    risks: list[AttackRisk] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def risk_of(self, attack: str) -> AttackRisk:
        """Return one attack's risk: `singling_out`, `linkability` or `inference:<column>`."""
        for entry in self.risks:
            if entry.attack == attack:
                return entry
        raise KeyError(attack)


def assess_attack_risks(
    original: pd.DataFrame,
    released: pd.DataFrame,
    *,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    control: pd.DataFrame | None = None,
    n_attacks: int = 500,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> AttackRiskReport:
    """Run the singling-out, linkability and inference attacks against a released table.

    `original` holds the real records and `released` what a recipient sees, both as pandas
    tables with the same columns. The attacker uses the quasi-identifiers, and for inference
    guesses each sensitive attribute from them. Singling out uses all of these columns.
    Linkability needs at least two quasi-identifiers, to split between the attacker's two
    databases. An attack that cannot run is listed in `skipped` with the reason.

    The number of attacks is cut to the number of records when the tables are smaller.

    Raises MissingExtraError (an ImportError) when the `anonymeter` extra is not installed.
    """
    evaluators = _load_anonymeter()
    # Linkability and inference draw each target record once, so no more attacks than records.
    sizes = [len(original), len(control) if control is not None else len(original)]
    n_attacks = min(n_attacks, *sizes)
    shared = [column for column in original.columns if column in released.columns]
    qi_columns = [column for column in quasi_identifiers if column in shared]
    secrets = [column for column in sensitive_attributes if column in shared]
    settings = _Settings(original, released, control, n_attacks, confidence_level, seed)
    risks, skipped = _run_attacks(evaluators, settings, qi_columns, secrets)
    return AttackRiskReport(confidence_level, n_attacks, risks, skipped)


@dataclass(frozen=True, slots=True)
class _Settings:
    original: pd.DataFrame
    released: pd.DataFrame
    control: pd.DataFrame | None
    n_attacks: int
    confidence_level: float
    seed: int | None


def _run_attacks(
    evaluators: Any, settings: _Settings, qi_columns: list[str], secrets: list[str]
) -> tuple[list[AttackRisk], dict[str, str]]:
    level = settings.confidence_level
    risks: list[AttackRisk] = []
    skipped: dict[str, str] = {}

    singling_columns = list(dict.fromkeys([*qi_columns, *secrets]))
    if singling_columns:
        outcome = _singling_out(evaluators, settings, singling_columns)
        risks.append(_risk_of(outcome, "singling_out", level))
    else:
        skipped["singling_out"] = "no quasi-identifier or sensitive column to query"

    if len(qi_columns) >= 2:
        outcome = _linkability(evaluators, settings, qi_columns)
        risks.append(_risk_of(outcome, "linkability", level))
    else:
        skipped["linkability"] = "needs at least two quasi-identifiers"

    for secret in secrets:
        known = [column for column in qi_columns if column != secret]
        if not known:
            skipped[f"inference:{secret}"] = "needs a quasi-identifier to guess from"
            continue
        outcome = _inference(evaluators, settings, known, secret)
        risks.append(_risk_of(outcome, f"inference:{secret}", level))
    return risks, skipped


def _load_anonymeter() -> Any:
    try:
        from anonymeter import evaluators  # type: ignore[import-untyped]
    except ImportError as error:
        raise MissingExtraError(
            'attack-based checks need the anonymeter extra: pip install "oblitus[anonymeter]"'
        ) from error
    return evaluators


def _risk_of(evaluator: Any, attack: str, confidence_level: float) -> AttackRisk:
    risk = evaluator.risk(confidence_level=confidence_level)
    low, high = risk.ci
    return AttackRisk(attack, float(risk.value), float(low), float(high))


def _singling_out(evaluators: Any, settings: _Settings, columns: list[str]) -> Any:
    evaluator = evaluators.SinglingOutEvaluator(
        ori=settings.original[columns],
        syn=settings.released[columns],
        control=_control_columns(settings, columns),
        n_attacks=settings.n_attacks,
        n_cols=min(_SINGLING_OUT_COLUMNS, len(columns)),
        max_attempts=_SINGLING_OUT_ATTEMPTS_PER_ATTACK * settings.n_attacks,
        seed=settings.seed,
    )
    return evaluator.evaluate()


def _linkability(evaluators: Any, settings: _Settings, columns: list[str]) -> Any:
    middle = len(columns) // 2
    evaluator = evaluators.LinkabilityEvaluator(
        ori=settings.original,
        syn=settings.released,
        control=settings.control,
        aux_cols=(columns[:middle], columns[middle:]),
        n_attacks=settings.n_attacks,
    )
    return evaluator.evaluate()


def _inference(evaluators: Any, settings: _Settings, known: list[str], secret: str) -> Any:
    columns = [*known, secret]
    evaluator = evaluators.InferenceEvaluator(
        ori=settings.original[columns],
        syn=settings.released[columns],
        control=_control_columns(settings, columns),
        aux_cols=known,
        secret=secret,
        regression=_is_continuous(settings.original[secret]),
        n_attacks=settings.n_attacks,
    )
    return evaluator.evaluate()


def _control_columns(settings: _Settings, columns: list[str]) -> pd.DataFrame | None:
    if settings.control is None:
        return None
    return settings.control[columns]


def _is_continuous(values: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(values) and values.nunique() > _REGRESSION_DISTINCT)
