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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
        return self


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


class Question(BaseModel):
    """A data-science question posed over the generated table, with a known answer.

    `target` is the column to predict. `hidden_columns` are generated, and may drive
    the target, but are withheld from the published data: a latent segment, or the
    true probability behind a label. `truth_column` names the hidden column holding
    the noise-free answer: the expected value for regression, or the probability of
    the positive class for classification. When it is left out for regression, the
    target's formula is evaluated again without its noise.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    prompt: str
    target: str
    task: Task
    hidden_columns: list[str] = Field(default_factory=list)
    truth_column: str | None = None
    test_share: float = Field(default=0.2, gt=0, lt=1)


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
            _check_question(self.question, set(by_name))
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


def _check_question(question: Question, names: set[str]) -> None:
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
