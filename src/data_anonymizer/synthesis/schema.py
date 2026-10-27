"""The schema that describes a table to generate: columns, kinds, ranges and choices.

A schema can be written by hand as JSON or inferred from a sample table. Every
field except `name` and `kind` is optional; which ones a column needs depends on
its kind (see `ColumnSchema`).
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_anonymizer.taxonomy import ColumnKind, DataCategory


class ColumnSchema(BaseModel):
    """How to generate one column.

    - integer / float: `minimum` and `maximum` for a uniform draw, or `mean` and `std`
      for a normal draw kept inside the optional bounds.
    - boolean: `true_share`, the probability of True (0.5 when left out).
    - category: `choices` with optional `weights`.
    - datetime: `start` and `end`.
    - text: `category` for realistic fakes (email, person_name, cpf, ...), or a
      `template` where `#` becomes a digit and `?` an uppercase letter, or `choices`.
    `category` also works for category columns, such as a small set of names.
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


class DatasetSchema(BaseModel):
    """An ordered list of column schemas with unique names."""

    model_config = ConfigDict(extra="forbid")

    columns: list[ColumnSchema] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> "DatasetSchema":
        names = [column.name for column in self.columns]
        duplicated = {name for name in names if names.count(name) > 1}
        if duplicated:
            raise ValueError(f"duplicated column names: {', '.join(sorted(duplicated))}")
        return self

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
