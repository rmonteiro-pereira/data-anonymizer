"""Turn a schema with a question into a data-science challenge with a known answer.

Because every pattern in the table was planted by the schema, the answer is not a
guess: the key lists the columns that truly drive the target, the ones that only
correlate with those drivers, the pure distractors, and the best score any model
can reach, measured against the noise-free truth.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from data_anonymizer.errors import SchemaError
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.expressions import evaluate, uses_randomness
from data_anonymizer.synthesis.generator import generate_from_schema
from data_anonymizer.synthesis.schema import DatasetSchema, Question, Task

# Probabilities are kept away from 0 and 1 so the log loss stays finite.
_PROBABILITY_EDGE = 1e-12


@dataclass(frozen=True, slots=True)
class Challenge:
    """The published files of a challenge and its private answer key."""

    train: pd.DataFrame
    test_features: pd.DataFrame
    test_answers: pd.DataFrame
    answer_key: dict[str, Any]


def build_challenge(
    schema: DatasetSchema,
    rows: int,
    *,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> Challenge:
    """Generate the table, split it, withhold hidden columns and write the answer key."""
    question = schema.question
    if question is None:
        raise SchemaError("the schema has no question to build a challenge from")

    full = generate_from_schema(schema, rows, rng=rng, surrogates=surrogates)
    truth = _truth(schema, question, full)
    in_test = rng.random(rows) < question.test_share
    published = full.drop(columns=question.hidden_columns)
    test_rows = published[in_test].reset_index(drop=True)

    answer_key = {
        "title": question.title,
        "prompt": question.prompt,
        "task": question.task.value,
        "target": question.target,
        "rows": {"train": int((~in_test).sum()), "test": int(in_test.sum())},
        "features": _feature_roles(schema, question),
        "how_the_target_was_built": _derivation(schema, question.target),
        "best_achievable": _best_scores(question, full[question.target][in_test], truth, in_test),
    }
    return Challenge(
        train=published[~in_test].reset_index(drop=True),
        test_features=test_rows.drop(columns=[question.target]),
        test_answers=test_rows[[question.target]],
        answer_key=answer_key,
    )


def _truth(schema: DatasetSchema, question: Question, full: pd.DataFrame) -> pd.Series | None:
    if question.truth_column is not None:
        return full[question.truth_column]
    target = schema.column(question.target)
    computable = (
        question.task is Task.REGRESSION
        and target.formula is not None
        and not uses_randomness(target.formula)
    )
    if computable:
        assert target.formula is not None
        return pd.Series(evaluate(target.formula, full).astype(float))
    return None


def _ancestors(schema: DatasetSchema, name: str) -> set[str]:
    found: set[str] = set()
    pending = list(schema.column(name).dependencies())
    while pending:
        current = pending.pop()
        if current not in found:
            found.add(current)
            pending.extend(schema.column(current).dependencies())
    return found


def _feature_roles(schema: DatasetSchema, question: Question) -> dict[str, list[str]]:
    hidden = set(question.hidden_columns)
    drivers = _ancestors(schema, question.target)
    correlated = {
        other
        for correlation in schema.correlations
        for first, other in (correlation.columns, correlation.columns[::-1])
        if first in drivers and other not in drivers
    }
    visible = [column.name for column in schema.columns if column.name not in hidden]
    candidates = [name for name in visible if name != question.target]
    return {
        "true_drivers": [name for name in candidates if name in drivers],
        "hidden_drivers": sorted(drivers & hidden),
        "correlated_but_not_causal": [name for name in candidates if name in correlated],
        "distractors": [
            name for name in candidates if name not in drivers and name not in correlated
        ],
    }


def _derivation(schema: DatasetSchema, target: str) -> list[dict[str, Any]]:
    involved = _ancestors(schema, target) | {target}
    steps = []
    for name in schema.generation_order():
        if name not in involved:
            continue
        column = schema.column(name)
        step: dict[str, Any] = {"column": name}
        if column.formula is not None:
            step["formula"] = column.formula
            if column.noise_std:
                step["noise_std"] = column.noise_std
        else:
            step["drawn_from"] = column.model_dump(
                exclude_none=True, exclude={"name", "conditions", "formula"}
            )
        if column.conditions:
            step["conditions"] = [
                override.model_dump(exclude_none=True) for override in column.conditions
            ]
        steps.append(step)
    return steps


def _best_scores(
    question: Question, answers: pd.Series, truth: pd.Series | None, in_test: np.ndarray
) -> dict[str, float] | str:
    if truth is None:
        return "not computable: name a truth_column to measure the noise-free optimum"
    best = truth[in_test].to_numpy(dtype=float)
    actual = answers.to_numpy(dtype=float)
    if question.task is Task.REGRESSION:
        residual = actual - best
        return {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1 - np.sum(residual**2) / np.sum((actual - actual.mean()) ** 2)),
        }
    return _classification_scores(actual, best)


def _classification_scores(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, _PROBABILITY_EDGE, 1 - _PROBABILITY_EDGE)
    log_loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    return {
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "roc_auc": _roc_auc(labels, probabilities),
        "log_loss": float(log_loss),
        "positive_rate": float(labels.mean()),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    # The AUC equals the Mann-Whitney U statistic scaled by the number of pairs.
    positives = labels == 1
    count_positive = int(positives.sum())
    count_negative = len(labels) - count_positive
    if count_positive == 0 or count_negative == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    rank_sum = ranks[positives].sum()
    return float(
        (rank_sum - count_positive * (count_positive + 1) / 2) / (count_positive * count_negative)
    )
