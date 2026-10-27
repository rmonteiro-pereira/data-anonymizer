"""Corrupt the published labels of a challenge, after the truth is known and rows are split.

Noise is applied to one split at a time, so the test answers stay clean unless the
question asks for noise on the test split.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oblitus.errors import SchemaError
from oblitus.synthesis.schema import ColumnSchema, LabelNoise, Question, Task


@dataclass(frozen=True, slots=True)
class NoisyLabels:
    """A label column after noise, and how many of its labels the noise changed."""

    labels: pd.Series
    changed: int


def corrupt_labels(
    labels: pd.Series,
    question: Question,
    noise: LabelNoise,
    target: ColumnSchema,
    rng: np.random.Generator,
) -> NoisyLabels:
    """Flip classification labels or add normal noise to regression labels."""
    if question.task is Task.CLASSIFICATION:
        assert noise.flip_rate is not None
        return _flip(labels, noise.flip_rate, target, rng)
    assert noise.std is not None
    return _perturb(labels, noise.std, rng)


def _flip(
    labels: pd.Series, flip_rate: float, target: ColumnSchema, rng: np.random.Generator
) -> NoisyLabels:
    # A missing label stays missing, so only present labels can flip.
    flips = (rng.random(len(labels)) < flip_rate) & labels.notna().to_numpy()
    opposite = _opposite_labels(labels, target)
    flipped = labels.where(~flips, opposite)
    return NoisyLabels(labels=flipped, changed=int(flips.sum()))


def _opposite_labels(labels: pd.Series, target: ColumnSchema) -> pd.Series:
    if target.choices:
        first, second = target.choices
        return labels.map({first: second, second: first})
    if pd.api.types.is_bool_dtype(labels.dtype):
        return ~labels
    if pd.api.types.is_numeric_dtype(labels.dtype):
        return 1 - labels
    raise SchemaError(f"cannot flip the labels of {target.name!r}: they are not boolean or 0/1")


def _perturb(labels: pd.Series, std: float, rng: np.random.Generator) -> NoisyLabels:
    noise = rng.normal(0.0, std, size=len(labels))
    perturbed = labels.astype("float64") + noise
    if pd.api.types.is_integer_dtype(labels.dtype):
        # An integer target stays integer, so the noise cannot reveal itself in the dtype.
        perturbed = perturbed.round().astype(labels.dtype)
    return NoisyLabels(labels=perturbed, changed=int(labels.notna().sum()))
