"""Difficulty levels: one setting on a question that stands for several concrete ones.

Each level sets four things; the numbers are in `PROFILES` and in docs/synthesis.md.
- Target noise multiplies the `noise_std` of the target's formula.
- Distractors are extra columns that nothing reads.
- Label noise flips classification labels, or adds noise to regression labels as a share of
  the target's standard deviation, in the train rows.
- Drift moves the mean of the first published numeric driver of the target in the test rows.

A level never overrides what the question already sets: its own `label_noise` or `test_shift`
is kept. A setting that does not fit the schema, such as noise on a target without `noise_std`
or drift on a panel, is skipped and the answer key says so.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from oblitus.synthesis.schema import (
    ColumnSchema,
    DatasetSchema,
    Difficulty,
    Distribution,
    LabelNoise,
    Question,
    ShiftedColumn,
    Task,
    columns_feeding,
)
from oblitus.taxonomy import ColumnKind


@dataclass(frozen=True, slots=True)
class DifficultyProfile:
    """The concrete settings behind one difficulty level."""

    noise_scale: float
    distractors: int
    flip_rate: float | None
    label_noise_std_share: float | None
    drift_in_std: float | None


PROFILES: dict[Difficulty, DifficultyProfile] = {
    Difficulty.EASY: DifficultyProfile(0.5, 0, None, None, None),
    Difficulty.MEDIUM: DifficultyProfile(1.0, 4, 0.02, 0.05, None),
    Difficulty.HARD: DifficultyProfile(2.0, 10, 0.05, 0.10, 0.5),
}

_DISTRACTOR_PREFIX = "extra_feature_"


@dataclass(frozen=True, slots=True)
class DifficultyApplied:
    """The schema after a difficulty level, and what the answer key says about it."""

    schema: DatasetSchema
    profile: DifficultyProfile
    notes: dict[str, Any]

    def with_label_noise(self, question: Question, train_target: pd.Series) -> Question:
        """Return the question with the level's label noise, unless it has noise of its own."""
        if question.label_noise is not None:
            return question
        noise = self._label_noise(question, train_target)
        if noise is None:
            return question
        self.notes["label_noise"] = noise.model_dump(mode="json", exclude_none=True)
        return question.model_copy(update={"label_noise": noise})

    def _label_noise(self, question: Question, train_target: pd.Series) -> LabelNoise | None:
        if question.task is Task.CLASSIFICATION:
            if self.profile.flip_rate is None or not _can_flip(self.schema, question):
                return None
            return LabelNoise(flip_rate=self.profile.flip_rate)
        share = self.profile.label_noise_std_share
        spread = float(pd.to_numeric(train_target, errors="coerce").std())
        if share is None or not spread > 0:
            return None
        return LabelNoise(std=share * spread)


def apply_difficulty(schema: DatasetSchema, *, supports_drift: bool = True) -> DifficultyApplied:
    """Apply the question's difficulty level to the schema, before any row is generated.

    Label noise needs the generated target, so `DifficultyApplied.with_label_noise` adds it later.
    """
    question = schema.question
    assert question is not None
    assert question.difficulty is not None
    profile = PROFILES[question.difficulty]
    notes: dict[str, Any] = {"level": question.difficulty.value}
    columns = _with_scaled_target_noise(schema, question, profile, notes)
    columns, added = _with_distractors(columns, profile.distractors)
    notes["distractor_columns"] = added
    shifts = _drift_shifts(schema, question, profile, supports_drift, notes)
    changed_question = question.model_copy(update={"test_shift": shifts})
    changed = schema.with_changes(columns=columns, question=changed_question)
    return DifficultyApplied(changed, profile, notes)


def _with_scaled_target_noise(
    schema: DatasetSchema, question: Question, profile: DifficultyProfile, notes: dict[str, Any]
) -> list[ColumnSchema]:
    target = schema.column(question.target)
    if not target.noise_std:
        notes["target_noise_std"] = "not applied: the target formula has no noise_std"
        return list(schema.columns)
    scaled = target.noise_std * profile.noise_scale
    notes["target_noise_std"] = {"before": target.noise_std, "after": scaled}
    updated = target.model_copy(update={"noise_std": scaled})
    return [updated if column.name == target.name else column for column in schema.columns]


def _with_distractors(
    columns: list[ColumnSchema], count: int
) -> tuple[list[ColumnSchema], list[str]]:
    taken = {column.name for column in columns}
    added: list[ColumnSchema] = []
    index = 0
    while len(added) < count:
        index += 1
        name = f"{_DISTRACTOR_PREFIX}{index:02d}"
        if name not in taken:
            added.append(_distractor(name, len(added)))
    return columns + added, [column.name for column in added]


def _distractor(name: str, position: int) -> ColumnSchema:
    # Cycle through a number, a category and a count so the extra columns are not all alike.
    shape = position % 3
    if shape == 0:
        return ColumnSchema(name=name, kind=ColumnKind.FLOAT, mean=0, std=1)
    if shape == 1:
        return ColumnSchema(name=name, kind=ColumnKind.CATEGORY, choices=["a", "b", "c", "d"])
    return ColumnSchema(
        name=name, kind=ColumnKind.INTEGER, distribution=Distribution.POISSON, mean=3
    )


def _can_flip(schema: DatasetSchema, question: Question) -> bool:
    target = schema.column(question.target)
    if target.choices:
        return len(target.choices) == 2
    return target.kind in (ColumnKind.BOOLEAN, ColumnKind.INTEGER, ColumnKind.FLOAT)


def _drift_shifts(
    schema: DatasetSchema,
    question: Question,
    profile: DifficultyProfile,
    supports_drift: bool,
    notes: dict[str, Any],
) -> list[ShiftedColumn]:
    if profile.drift_in_std is None:
        return list(question.test_shift)
    if question.test_shift:
        return list(question.test_shift)
    driver = _first_shiftable_driver(schema, question) if supports_drift else None
    if driver is None:
        notes["drift"] = "not applied: no published numeric driver with a mean and std to shift"
        return []
    assert driver.mean is not None and driver.std is not None
    new_mean = driver.mean + profile.drift_in_std * driver.std
    notes["drift"] = {"column": driver.name, "mean_before": driver.mean, "mean_after": new_mean}
    return [ShiftedColumn(name=driver.name, mean=new_mean)]


def _first_shiftable_driver(schema: DatasetSchema, question: Question) -> ColumnSchema | None:
    by_name = {column.name: column for column in schema.columns}
    drivers = columns_feeding(by_name, question.target)
    hidden = set(question.hidden_columns)
    for column in schema.columns:
        eligible = (
            column.name in drivers
            and column.name not in hidden
            and column.formula is None
            and column.kind in (ColumnKind.INTEGER, ColumnKind.FLOAT)
            and column.mean is not None
            and column.std is not None
            and not column.unique
            and schema.panel is None
        )
        if eligible:
            return column
    return None
