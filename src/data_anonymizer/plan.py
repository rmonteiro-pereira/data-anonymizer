"""An anonymization plan: which technique, with which parameters, runs on which column.

Plans are plain JSON so a person can review and edit a proposal before it runs.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Fidelity(StrEnum):
    """What the plan protects first when privacy and analytic usefulness conflict.

    `statistical` keeps distributions, correlations, distinct counts and joins as close
    to the original as it can, accepting that the result is pseudonymous rather than
    anonymous. `privacy` generalizes, suppresses and randomizes so records are hard
    to single out, accepting distorted statistics.
    """

    STATISTICAL = "statistical"
    PRIVACY = "privacy"


class ColumnAction(BaseModel):
    """One technique applied to one column, with the reason it was chosen."""

    model_config = ConfigDict(extra="forbid")

    column: str
    technique: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class AnonymizationPlan(BaseModel):
    """The column actions, plus the quasi-identifiers and sensitive attributes they protect.

    With `enforce_k_anonymity`, after the column actions run, quasi-identifier cells
    are blanked in rows whose combination is shared by fewer than `k_target` rows.
    """

    model_config = ConfigDict(extra="forbid")

    fidelity: Fidelity
    actions: list[ColumnAction]
    quasi_identifiers: list[str] = Field(default_factory=list)
    sensitive_attributes: list[str] = Field(default_factory=list)
    k_target: int = Field(default=5, ge=1)
    enforce_k_anonymity: bool = False

    @model_validator(mode="after")
    def _one_action_per_column(self) -> "AnonymizationPlan":
        seen: set[str] = set()
        for action in self.actions:
            if action.column in seen:
                raise ValueError(f"column {action.column!r} has more than one action")
            seen.add(action.column)
        return self

    def action_for(self, column: str) -> ColumnAction | None:
        """Return the action planned for a column, or None when it is left alone."""
        for action in self.actions:
            if action.column == column:
                return action
        return None

    def to_json(self) -> str:
        """Serialize the plan as indented JSON for review."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "AnonymizationPlan":
        """Parse and validate a plan from JSON."""
        return cls.model_validate_json(text)

    def save(self, path: Path) -> None:
        """Write the plan to a JSON file."""
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AnonymizationPlan":
        """Read a plan from a JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))
