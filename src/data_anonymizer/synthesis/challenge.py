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
from data_anonymizer.synthesis.causal_graph import CausalGraph
from data_anonymizer.synthesis.expressions import evaluate, uses_randomness
from data_anonymizer.synthesis.generator import generate_with_replay
from data_anonymizer.synthesis.schema import ColumnSchema, DatasetSchema, Question, Task
from data_anonymizer.synthesis.treatment_effects import causal_answer
from data_anonymizer.taxonomy import ColumnKind

# Probabilities are kept away from 0 and 1 so the log loss stays finite.
_PROBABILITY_EDGE = 1e-12

_NAME_A_TRUTH_COLUMN = "name a truth_column to measure the noise-free optimum"


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

    built = generate_with_replay(schema, rows, rng=rng, surrogates=surrogates)
    full = built.table
    truth = _truth(schema, question, full)
    in_test = _split_rows(question, rows, rng)
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
    train = published[~in_test].reset_index(drop=True)
    if question.treatment is not None:

        def outcome_as_numbers(answers: pd.Series) -> np.ndarray:
            return _numeric_answers(question, answers)

        answer_key["causal"] = causal_answer(schema, question, built, train, outcome_as_numbers)
    return Challenge(
        train=train,
        test_features=test_rows.drop(columns=[question.target]),
        test_answers=test_rows[[question.target]],
        answer_key=answer_key,
    )


def _split_rows(question: Question, rows: int, rng: np.random.Generator) -> np.ndarray:
    in_test = rng.random(rows) < question.test_share
    test_rows = int(in_test.sum())
    train_rows = rows - test_rows
    if train_rows == 0 or test_rows == 0:
        raise SchemaError(
            f"{rows} rows leave {train_rows} training rows and {test_rows} test rows; "
            "ask for more rows or change the question's test_share"
        )
    return in_test


def _truth(schema: DatasetSchema, question: Question, full: pd.DataFrame) -> pd.Series | str:
    """Return the noise-free answer per row, or the reason it cannot be computed."""
    if question.truth_column is not None:
        return full[question.truth_column]
    target = schema.column(question.target)
    no_formula = target.formula is None or uses_randomness(target.formula)
    if question.task is not Task.REGRESSION or no_formula:
        return f"not computable: {_NAME_A_TRUTH_COLUMN}"
    assert target.formula is not None
    changes = _changes_after_formula(target)
    if changes:
        return (
            f"not computable: {', '.join(changes)} change the target after its formula, "
            f"so the formula is not its noise-free truth; {_NAME_A_TRUTH_COLUMN}"
        )
    return pd.Series(evaluate(target.formula, _with_float_nulls(full)).astype(float))


def _changes_after_formula(target: ColumnSchema) -> list[str]:
    changes = []
    if target.minimum is not None or target.maximum is not None:
        changes.append("bounds")
    if target.exact:
        changes.append("exact statistics")
    if target.conditions:
        changes.append("conditions")
    if target.outlier_fraction > 0:
        changes.append("outliers")
    if target.kind is ColumnKind.INTEGER:
        changes.append("integer rounding")
    return changes


def _with_float_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    # Nullable integer and boolean columns hold NA objects that arithmetic cannot use.
    plain = frame.copy()
    for name in plain.columns:
        dtype = plain[name].dtype
        is_nullable_number = isinstance(dtype, pd.api.extensions.ExtensionDtype) and (
            pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
        )
        if is_nullable_number:
            plain[name] = plain[name].astype("float64")
    return plain


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
    leaks = CausalGraph.of(schema).descendants(question.target)
    candidates = [name for name in visible if name != question.target and name not in leaks]
    return {
        "true_drivers": [name for name in candidates if name in drivers],
        "hidden_drivers": sorted(drivers & hidden),
        "correlated_but_not_causal": [name for name in candidates if name in correlated],
        "leaks_target": [name for name in visible if name in leaks],
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
    question: Question, answers: pd.Series, truth: pd.Series | str, in_test: np.ndarray
) -> dict[str, float] | str:
    if isinstance(truth, str):
        return truth
    actual = _numeric_answers(question, answers)
    best = truth[in_test].to_numpy(dtype=float, na_value=np.nan)
    # A row with no label or no truth cannot be scored.
    scored = ~np.isnan(actual) & ~np.isnan(best)
    if not scored.any():
        return "not computable: no test row has both a target value and a truth value"
    actual = actual[scored]
    best = best[scored]
    if question.task is Task.REGRESSION:
        residual = actual - best
        return {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1 - np.sum(residual**2) / np.sum((actual - actual.mean()) ** 2)),
        }
    return _classification_scores(actual, best)


def _numeric_answers(question: Question, answers: pd.Series) -> np.ndarray:
    """Return the target as numbers, with NaN for missing cells."""
    if question.task is Task.CLASSIFICATION and question.positive_label is not None:
        labels = (answers.astype(str) == question.positive_label).to_numpy(dtype=float)
        labels[answers.isna().to_numpy()] = np.nan
        return labels
    try:
        numbers = answers.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise SchemaError(
            f"the target {question.target!r} is not numeric; "
            "a classification question can name a positive_label instead"
        ) from error
    present = numbers[~np.isnan(numbers)]
    if question.task is Task.CLASSIFICATION and not np.isin(present, [0.0, 1.0]).all():
        raise SchemaError(
            f"the classification target {question.target!r} is not boolean or 0/1; "
            "name its positive class in the question's positive_label"
        )
    return numbers


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
