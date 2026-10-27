"""The schema that describes a table to generate: columns, distributions and patterns.

A schema can be written by hand as JSON or inferred from a sample table. Every
field except `name` and `kind` is optional; which ones a column needs depends on
its kind (see `ColumnSchema`). Beyond single columns, a schema can fix rank
correlations between columns, conditional rules ("when plan is pro, spend is
higher"), derived columns computed by formula, and constraints every row obeys.
"""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_anonymizer.synthesis.causal_graph import CausalGraph
from data_anonymizer.synthesis.draw_replay import DRAW_PREFIX
from data_anonymizer.synthesis.expressions import parse, referenced_columns, uses_randomness
from data_anonymizer.taxonomy import ColumnKind, DataCategory

# Kinds whose values come from an inverse distribution function, so they can be
# correlated with each other and redrawn under conditions.
DRAWN_KINDS = frozenset(
    {
        ColumnKind.INTEGER,
        ColumnKind.FLOAT,
        ColumnKind.BOOLEAN,
        ColumnKind.CATEGORY,
        ColumnKind.DATETIME,
    }
)


class Distribution(StrEnum):
    """The shape of a numeric column.

    normal, lognormal and gamma take `mean` and `std`; exponential and poisson take
    `mean`; uniform takes `minimum` and `maximum`; quantiles takes `quantiles`, a map
    from probability to value such as {"0": 0, "0.5": 3000, "0.9": 9000, "1": 50000},
    joined by straight lines. `minimum` and `maximum` truncate every shape.
    """

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    UNIFORM = "uniform"
    QUANTILES = "quantiles"


# Fields of an override that select rows or blank cells rather than change the distribution.
_NOT_DISTRIBUTION_FIELDS = {"when", "null_fraction"}


class ColumnOverride(BaseModel):
    """Different distribution settings for the rows where `when` is true.

    `when` is an expression over other columns, such as `plan == 'pro'` or
    `age >= 60 and city in ['Recife', 'Natal']`. Rows keep their position in the
    distribution, so correlations survive the override.
    """

    model_config = ConfigDict(extra="forbid")

    when: str
    distribution: Distribution | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = Field(default=None, ge=0)
    quantiles: dict[float, float] | None = None
    true_share: float | None = Field(default=None, ge=0, le=1)
    choices: list[str] | None = None
    weights: list[float] | None = None
    start: datetime | None = None
    end: datetime | None = None
    null_fraction: float | None = Field(default=None, ge=0, le=1)

    @field_validator("when")
    @classmethod
    def _parseable(cls, text: str) -> str:
        parse(text)
        return text


class ColumnSchema(BaseModel):
    """How to generate one column.

    - integer / float: a `distribution` with its parameters. Left out, it is normal
      when `mean` and `std` are given, quantiles when `quantiles` is, and uniform
      between `minimum` and `maximum` otherwise.
    - boolean: `true_share`, the probability of True (0.5 when left out).
    - category: `choices` with optional `weights`.
    - datetime: `start` and `end`.
    - text: `category` for realistic fakes (email, person_name, cpf, ...), or a
      `template` where `#` becomes a digit and `?` an uppercase letter, or `choices`.
    `category` also works for category columns, such as a small set of names.

    Patterns:
    - `formula`: compute the column from others, such as `50 + 2.5 * age`, plus
      normal noise of `noise_std`. Date functions (`dayofyear`, `days`, ...) and
      `sin` express trends and seasonality.
    - `conditions`: overrides that apply where their `when` expression is true.
    - `outlier_fraction`: share of rows pushed `outlier_scale` standard deviations
      away from the mean, ignoring the bounds, to plant anomalies.
    - `exact`: match `mean` and `std` (numbers) or `weights` (categories, booleans)
      exactly rather than on average.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: ColumnKind
    category: DataCategory | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = Field(default=None, ge=0)
    true_share: float | None = Field(default=None, ge=0, le=1)
    choices: list[str] | None = None
    weights: list[float] | None = None
    start: datetime | None = None
    end: datetime | None = None
    template: str | None = None
    null_fraction: float = Field(default=0.0, ge=0, le=1)
    unique: bool = False
    distribution: Distribution | None = None
    quantiles: dict[float, float] | None = None
    formula: str | None = None
    noise_std: float | None = Field(default=None, ge=0)
    conditions: list[ColumnOverride] = Field(default_factory=list)
    outlier_fraction: float = Field(default=0.0, ge=0, le=1)
    outlier_scale: float = Field(default=6.0, gt=0)
    exact: bool = False

    @field_validator("formula")
    @classmethod
    def _parseable(cls, text: str | None) -> str | None:
        if text is not None:
            parse(text)
        return text

    def dependencies(self) -> set[str]:
        """Return the other columns this column's formula and conditions read."""
        names: set[str] = set()
        if self.formula is not None:
            names |= referenced_columns(self.formula)
        for override in self.conditions:
            names |= referenced_columns(override.when)
        return names - {self.name}

    def with_override(self, override: ColumnOverride) -> "ColumnSchema":
        """Return this column with an override's settings applied, validated as a whole."""
        changes = {
            field: value
            for field, value in override.model_dump().items()
            if field not in _NOT_DISTRIBUTION_FIELDS and value is not None
        }
        return self._with_changes(changes, keeps_patterns=False)

    def with_shift(self, shift: "ShiftedColumn") -> "ColumnSchema":
        """Return this column with a test-set shift merged in, validated as a whole.

        Unlike an override, a shift keeps the column's formula and conditions unless it
        names new ones, and it can change the null fraction.
        """
        changes = {
            field: value
            for field, value in shift.model_dump(exclude={"name", "when"}).items()
            if value is not None
        }
        return self._with_changes(changes, keeps_patterns=True)

    def _with_changes(self, changes: dict[str, Any], *, keeps_patterns: bool) -> "ColumnSchema":
        resets_shape = self._override_replaces_shape(changes)
        if resets_shape:
            # A uniform or quantile shape would ignore the new mean and std.
            changes.update(distribution=None, quantiles=None)
        if "quantiles" in changes and "distribution" not in changes:
            changes["distribution"] = Distribution.QUANTILES
        if "choices" in changes and "weights" not in changes:
            # The base weights belong to the base choices.
            changes["weights"] = None
        settings = {**self.model_dump(), **changes}
        if not keeps_patterns:
            settings.update(conditions=[], formula=None)
        merged = ColumnSchema.model_validate(settings)
        if resets_shape and (merged.mean is None or merged.std is None):
            raise ValueError(
                f"{self.name}: an override of mean or std on a uniform or quantile column "
                "needs both mean and std"
            )
        return merged

    def _override_replaces_shape(self, changes: dict[str, Any]) -> bool:
        sets_spread = "mean" in changes or "std" in changes
        sets_shape = "distribution" in changes or "quantiles" in changes
        has_fixed_shape = self.quantiles is not None or self.distribution in (
            Distribution.UNIFORM,
            Distribution.QUANTILES,
        )
        return sets_spread and not sets_shape and has_fixed_shape

    @model_validator(mode="after")
    def _consistent_fields(self) -> "ColumnSchema":
        if self.weights is not None:
            if self.choices is None or len(self.weights) != len(self.choices):
                raise ValueError(f"{self.name}: weights need one entry per choice")
            if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
                raise ValueError(f"{self.name}: weights must be non-negative and not all zero")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"{self.name}: minimum is greater than maximum")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"{self.name}: start is after end")
        self._check_unique_is_possible()
        for override in self.conditions:
            self.with_override(override)
        return self

    def _check_unique_is_possible(self) -> None:
        if not self.unique:
            return
        # Booleans and dates have no room for many distinct values, and a list of choices
        # runs out. Keys, fake values, templates and numbers can make fresh ones.
        if self.kind in (ColumnKind.BOOLEAN, ColumnKind.DATETIME) or self.choices:
            raise ValueError(
                f"{self.name}: unique is not possible for {self.kind.value} columns or "
                "columns with choices; use an integer key, a fake-value category or a template"
            )


class Correlation(BaseModel):
    """A Spearman rank correlation, between -1 and 1, that two columns must show.

    Category columns are ranked in the order of their `choices`, and booleans rank
    False below True.
    """

    model_config = ConfigDict(extra="forbid")

    columns: tuple[str, str]
    value: float = Field(ge=-1, le=1)


class Task(StrEnum):
    """The kind of prediction a generated challenge asks for."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ShiftedColumn(ColumnOverride):
    """New settings for one column of the test set, which stays the same in the train set.

    It takes every setting of an override, plus a new `formula`, `noise_std` and `conditions`
    that replace the column's own. It applies to the whole test set, so it has no `when`:
    put a `when` in `conditions` to shift only some rows.
    """

    name: str
    when: str | None = None  # type: ignore[assignment]
    formula: str | None = None
    noise_std: float | None = Field(default=None, ge=0)
    conditions: list[ColumnOverride] | None = None

    @field_validator("formula")
    @classmethod
    def _parseable_formula(cls, text: str | None) -> str | None:
        if text is not None:
            parse(text)
        return text

    @model_validator(mode="after")
    def _has_no_row_filter(self) -> "ShiftedColumn":
        if self.when is not None:
            raise ValueError(
                f"test_shift {self.name}: a shift applies to the whole test set; "
                "put the row filter in `conditions`"
            )
        return self

    def changes_mechanism(self) -> bool:
        """Return whether the shift replaces how the column is computed, not just its draw."""
        return self.formula is not None or self.noise_std is not None or self.conditions is not None


class NoiseSplit(StrEnum):
    """Which published labels receive label noise."""

    TRAIN = "train"
    TEST = "test"
    BOTH = "both"


class LabelNoise(BaseModel):
    """Corruption of the published labels, after the truth is known and the rows are split.

    A classification target is flipped with probability `flip_rate`. A regression target
    gets normal noise of standard deviation `std`. The `split` says where it applies.
    """

    model_config = ConfigDict(extra="forbid")

    flip_rate: float | None = Field(default=None, gt=0, le=0.5)
    std: float | None = Field(default=None, gt=0)
    split: NoiseSplit = NoiseSplit.TRAIN

    @model_validator(mode="after")
    def _one_kind_of_noise(self) -> "LabelNoise":
        if (self.flip_rate is None) == (self.std is None):
            raise ValueError("label_noise needs exactly one of flip_rate and std")
        return self

    def applies_to(self, split: str) -> bool:
        """Return whether the named split ("train" or "test") is corrupted."""
        return self.split is NoiseSplit.BOTH or self.split.value == split


class Question(BaseModel):
    """A data-science question posed over the generated table, with a known answer.

    `target` is the column to predict. `hidden_columns` are generated, and may drive
    the target, but are withheld from the published data: a latent segment, or the
    true probability behind a label. `truth_column` names the hidden column holding
    the noise-free answer: the expected value for regression, or the probability of
    the positive class for classification. When it is left out for regression, the
    target's formula is evaluated again without its noise.

    `test_shift` changes columns in the test rows only, so the test set drifts away from the
    train set. `label_noise` corrupts the published labels of the chosen split.

    A classification target is boolean or 0/1. Any other target, such as a category with
    choices "no" and "yes", names its positive class in `positive_label`.

    A causal question names a `treatment` column. `treatment_values` gives the control value
    first and the treated value second; it may be left out for a boolean column (False, True)
    and for an integer column bounded to 0 and 1. `effect_modifiers` are columns, not caused
    by the treatment, to report conditional effects by. The answer key then holds the true
    effects, found by setting the treatment for every row and replaying the other random numbers.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    prompt: str
    target: str
    task: Task
    hidden_columns: list[str] = Field(default_factory=list)
    truth_column: str | None = None
    positive_label: str | None = None
    treatment: str | None = None
    treatment_values: list[Any] | None = None
    effect_modifiers: list[str] = Field(default_factory=list)
    test_share: float = Field(default=0.2, gt=0, lt=1)
    test_shift: list[ShiftedColumn] = Field(default_factory=list)
    label_noise: LabelNoise | None = None


class DatasetSchema(BaseModel):
    """Column schemas with unique names, plus correlations and row constraints.

    `constraints` are expressions every generated row must satisfy, such as
    `end_date >= start_date`; rows that break one are drawn again.
    """

    model_config = ConfigDict(extra="forbid")

    columns: list[ColumnSchema] = Field(min_length=1)
    correlations: list[Correlation] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    question: Question | None = None

    @model_validator(mode="after")
    def _unique_names(self) -> "DatasetSchema":
        names = [column.name for column in self.columns]
        duplicated = {name for name in names if names.count(name) > 1}
        if duplicated:
            raise ValueError(f"duplicated column names: {', '.join(sorted(duplicated))}")
        reserved = [name for name in names if name.startswith(DRAW_PREFIX)]
        if reserved:
            raise ValueError(f"column names cannot start with {DRAW_PREFIX!r}: {reserved}")
        return self

    @model_validator(mode="after")
    def _patterns_refer_to_known_columns(self) -> "DatasetSchema":
        by_name = {column.name: column for column in self.columns}
        for column in self.columns:
            unknown = column.dependencies() - set(by_name)
            if unknown:
                raise ValueError(f"{column.name}: unknown columns {', '.join(sorted(unknown))}")
        for correlation in self.correlations:
            for name in correlation.columns:
                _check_correlatable(by_name.get(name), name)
        for expression in self._checked_expressions():
            if uses_randomness(expression):
                raise ValueError(f"{expression!r}: random functions are only allowed in formulas")
        for constraint in self.constraints:
            unknown = referenced_columns(constraint) - set(by_name)
            if unknown:
                raise ValueError(f"constraint {constraint!r}: unknown columns {sorted(unknown)}")
        self.generation_order()
        if self.question is not None:
            _check_question(self.question, by_name)
            self.shifted(self.question.test_shift)
        return self

    def _checked_expressions(self) -> list[str]:
        conditions = [override.when for column in self.columns for override in column.conditions]
        return conditions + list(self.constraints)

    def column(self, name: str) -> ColumnSchema:
        """Return the schema of one column."""
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def shifted(self, shifts: list[ShiftedColumn]) -> "DatasetSchema":
        """Return the schema with test-set shifts merged into the named columns.

        The result has no question, and is validated as a whole like any schema.
        """
        replacements = {}
        for shift in shifts:
            if shift.name in replacements:
                raise ValueError(f"test_shift names {shift.name!r} twice")
            try:
                replacements[shift.name] = self.column(shift.name).with_shift(shift)
            except KeyError:
                raise ValueError(f"test_shift names unknown column {shift.name!r}") from None
        columns = [replacements.get(column.name, column) for column in self.columns]
        return DatasetSchema(
            columns=columns, correlations=self.correlations, constraints=self.constraints
        )

    def generation_order(self) -> list[str]:
        """Order columns so each comes after the columns its formula and conditions read."""
        dependencies = {column.name: column.dependencies() for column in self.columns}
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in ordered:
                return
            if name in visiting:
                raise ValueError(f"columns depend on each other in a cycle through {name!r}")
            visiting.add(name)
            for dependency in sorted(dependencies[name]):
                visit(dependency)
            visiting.discard(name)
            ordered.append(name)

        for column in self.columns:
            visit(column.name)
        return ordered

    def to_json(self) -> str:
        """Serialize the schema as indented JSON, leaving out empty fields."""
        return self.model_dump_json(indent=2, exclude_none=True)

    @classmethod
    def from_json(cls, text: str) -> "DatasetSchema":
        """Parse and validate a schema from JSON."""
        return cls.model_validate_json(text)

    def save(self, path: Path) -> None:
        """Write the schema to a JSON file."""
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DatasetSchema":
        """Read a schema from a JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))


def _check_correlatable(column: ColumnSchema | None, name: str) -> None:
    if column is None:
        raise ValueError(f"correlation names unknown column {name!r}")
    drawn = column.kind in DRAWN_KINDS and column.category is None
    if not drawn or column.formula is not None or column.unique:
        raise ValueError(
            f"correlation column {name!r} must be a drawn number, date, boolean or "
            "category, not a formula, unique key or fake-value column"
        )


def _check_question(question: Question, by_name: dict[str, ColumnSchema]) -> None:
    names = set(by_name)
    mentioned = {question.target, *question.hidden_columns}
    if question.truth_column is not None:
        mentioned.add(question.truth_column)
        if question.truth_column not in question.hidden_columns:
            raise ValueError("the question's truth_column must also be a hidden column")
    unknown = mentioned - names
    if unknown:
        raise ValueError(f"the question names unknown columns {', '.join(sorted(unknown))}")
    if question.target in question.hidden_columns:
        raise ValueError("the question's target cannot be hidden")
    _check_positive_label(question, by_name[question.target])
    _check_label_noise(question, by_name[question.target])
    _check_treatment(question, by_name)


def _check_positive_label(question: Question, target: ColumnSchema) -> None:
    if question.task is not Task.CLASSIFICATION:
        return
    is_binary_kind = target.kind in (ColumnKind.BOOLEAN, ColumnKind.INTEGER, ColumnKind.FLOAT)
    if question.positive_label is None and not is_binary_kind:
        raise ValueError(
            f"the classification target {target.name!r} is not boolean or 0/1, "
            "so the question needs a positive_label"
        )
    unknown_label = (
        question.positive_label is not None
        and bool(target.choices)
        and question.positive_label not in (target.choices or [])
    )
    if unknown_label:
        raise ValueError(
            f"positive_label {question.positive_label!r} is not one of the target's choices"
        )


def _check_label_noise(question: Question, target: ColumnSchema) -> None:
    noise = question.label_noise
    if noise is None:
        return
    if question.task is Task.CLASSIFICATION and noise.flip_rate is None:
        raise ValueError("label_noise on a classification question needs a flip_rate")
    if question.task is Task.REGRESSION and noise.std is None:
        raise ValueError("label_noise on a regression question needs a std")
    has_choices = target.kind is ColumnKind.CATEGORY or bool(target.choices)
    needs_two_choices = question.task is Task.CLASSIFICATION and has_choices
    if needs_two_choices and len(target.choices or []) != 2:
        raise ValueError(
            f"label_noise flips the labels of {target.name!r}, which needs exactly two choices"
        )


def treatment_levels(question: Question, treatment: ColumnSchema) -> tuple[Any, Any]:
    """Return the control value and the treated value of a causal question."""
    if question.treatment_values is not None:
        return question.treatment_values[0], question.treatment_values[1]
    if treatment.kind is ColumnKind.BOOLEAN:
        return False, True
    if treatment.kind is ColumnKind.INTEGER and (treatment.minimum, treatment.maximum) == (0, 1):
        return 0, 1
    raise ValueError(
        f"the treatment {treatment.name!r} is not boolean or an integer bounded to 0 and 1, "
        "so the question needs treatment_values: [control, treated]"
    )


def _check_treatment(question: Question, by_name: dict[str, ColumnSchema]) -> None:
    if question.treatment is None:
        if question.treatment_values is not None or question.effect_modifiers:
            raise ValueError("treatment_values and effect_modifiers need a treatment")
        return
    treatment = question.treatment
    if treatment not in by_name:
        raise ValueError(f"the question's treatment {treatment!r} is not a column of the schema")
    if treatment == question.target:
        raise ValueError("the question's treatment cannot be its target")
    if treatment in question.hidden_columns:
        raise ValueError("the question's treatment cannot be hidden")
    graph = CausalGraph.from_columns(by_name.values())
    if question.target in graph.ancestors(treatment):
        raise ValueError(
            f"the treatment {treatment!r} is caused by the target {question.target!r}, "
            "so it cannot be a treatment for it"
        )
    _check_treatment_values(question, by_name[treatment])
    _check_effect_modifiers(question, by_name, graph)


def _check_treatment_values(question: Question, treatment: ColumnSchema) -> None:
    if question.treatment_values is not None and len(question.treatment_values) != 2:
        raise ValueError("treatment_values needs two values: the control, then the treated")
    control, treated = treatment_levels(question, treatment)
    if control == treated:
        raise ValueError("the control and treated values of the treatment must differ")
    for value in (control, treated):
        if treatment.choices and value not in treatment.choices:
            raise ValueError(f"treatment value {value!r} is not a choice of {treatment.name!r}")


def _check_effect_modifiers(
    question: Question, by_name: dict[str, ColumnSchema], graph: CausalGraph
) -> None:
    assert question.treatment is not None
    after_treatment = graph.descendants(question.treatment)
    reportable = (ColumnKind.BOOLEAN, ColumnKind.CATEGORY, ColumnKind.INTEGER, ColumnKind.FLOAT)
    for name in question.effect_modifiers:
        if name not in by_name:
            raise ValueError(f"the effect modifier {name!r} is not a column of the schema")
        if name in (question.treatment, question.target) or name in after_treatment:
            raise ValueError(
                f"the effect modifier {name!r} must not be the treatment, the target "
                "or caused by the treatment"
            )
        if by_name[name].kind not in reportable:
            raise ValueError(f"the effect modifier {name!r} must be a boolean, category or number")
