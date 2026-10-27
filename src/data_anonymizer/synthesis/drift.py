"""Shift the test rows of a challenge away from the train rows.

The test rows are generated again from the schema with the question's shifts merged into
the named columns. The train rows keep the base schema.
"""

import numpy as np
import pandas as pd

from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.generator import generate_from_schema
from data_anonymizer.synthesis.schema import DatasetSchema, Question


def with_shifted_test_rows(
    shifted_schema: DatasetSchema,
    table: pd.DataFrame,
    in_test: np.ndarray,
    *,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace the test rows with rows from the shifted schema.

    The train rows come first in the result, followed by the test rows, and the returned
    mask marks the test rows. Unique keys are unique within each split, not across them.
    """
    train_rows = table[~in_test]
    test_count = int(in_test.sum())
    test_rows = generate_from_schema(shifted_schema, test_count, rng=rng, surrogates=surrogates)
    combined = pd.concat([train_rows, test_rows], ignore_index=True)
    mask = np.arange(len(combined)) >= len(train_rows)
    return combined, mask


def drift_kind(question: Question, target_ancestors: set[str]) -> str:
    """Name the drift: "concept" when the target's mechanism changes, else "covariate".

    The mechanism changes when the target is shifted, or when a column it is computed from
    gets a new formula, noise or conditions. Only new distributions of drivers, or of
    columns that do not drive the target, move the inputs and leave the mechanism alone.
    """
    for shift in question.test_shift:
        if shift.name == question.target:
            return "concept"
        if shift.name in target_ancestors and shift.changes_mechanism():
            return "concept"
    return "covariate"


def describe_drift(
    question: Question, kind: str, best_achievable: dict[str, float] | str
) -> dict[str, object]:
    """Return the answer key's drift section."""
    return {
        "shifted_columns": [shift.name for shift in question.test_shift],
        "kind": kind,
        "shifts": [shift.model_dump(exclude_none=True) for shift in question.test_shift],
        "best_achievable": best_achievable,
    }
