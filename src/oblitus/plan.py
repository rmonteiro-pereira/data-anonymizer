"""An anonymization plan: which technique, with which parameters, runs on which column.

Plans are plain JSON so a person can review and edit a proposal before it runs.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oblitus.generalization_hierarchies import GeneralizationHierarchy


class Fidelity(StrEnum):
    """What the plan protects first when privacy and analytic usefulness conflict.

    `statistical` keeps distributions, correlations, distinct counts and joins as close
    to the original as it can, accepting that the result is pseudonymous rather than
    anonymous. `privacy` generalizes, suppresses and randomizes so records are hard
    to single out, accepting distorted statistics.
    """

    STATISTICAL = "statistical"
    PRIVACY = "privacy"


class Protection(StrEnum):
    """Whether the result may still be tied back to people by whoever holds the key.

    `pseudonymization` replaces identifiers with keyed or encrypted stand-ins: the
    key holder can re-link or reveal records, so under GDPR and LGPD the output is
    still personal data. `anonymization` removes every link: no keyed, consistent
    or reversible technique is allowed, quasi-identifiers are generalized until each
    combination is shared by at least `k_target` rows, and the result is checked.
    """

    PSEUDONYMIZATION = "pseudonymization"
    ANONYMIZATION = "anonymization"


class KAnonymityMethod(StrEnum):
    """How `enforce_k_anonymity` makes every quasi-identifier combination shared by k rows.

    `local_suppression` blanks the quasi-identifier cells of rows in small groups and
    changes nothing else. `mondrian` cuts the rows into groups of at least k on the
    quasi-identifiers and replaces every cell with its group's range or set of values, so no
    cell is blank but every quasi-identifier column becomes coarser.
    """

    LOCAL_SUPPRESSION = "local_suppression"
    MONDRIAN = "mondrian"


class RangeStyle(StrEnum):
    """How Mondrian writes the numbers and dates of a group: its `lo-hi` range or its midpoint."""

    RANGE = "range"
    MIDPOINT = "midpoint"


class JointMicroaggregation(BaseModel):
    """Several numeric columns replaced together, in groups of at least `group_size` records.

    Records are grouped by their distance over all the columns at once (the MDAV method), and
    each column then holds its group's mean. Every released vector is shared by at least
    `group_size` records.
    """

    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(min_length=2)
    group_size: int = Field(default=3, ge=2)

    @model_validator(mode="after")
    def _columns_are_distinct(self) -> "JointMicroaggregation":
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("a column is listed more than once")
        return self


class ColumnAction(BaseModel):
    """One technique applied to one column, with the reason it was chosen."""

    model_config = ConfigDict(extra="forbid")

    column: str
    technique: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class AnonymizationPlan(BaseModel):
    """The column actions, plus the identifiers and attributes they protect.

    `entity_column` names the column that identifies whose row it is (a person, a
    store), for techniques that treat all rows of one entity alike.

    `row_key` names a column that identifies each row, such as a record ID. Random
    numbers per row are keyed on it, so the same row gets the same noise on every engine
    and every run, and a DuckDB table can be lined up with its released version. In-memory
    tables without one use the row position; DuckDB tables must name one, and it must be
    unique.

    With `enforce_k_anonymity`, after the column actions run, quasi-identifier cells
    are blanked in rows whose combination is shared by fewer than `k_target` rows.
    `l_target` is the fewest distinct sensitive values each such group must hold for
    the result to count as anonymized.

    `k_anonymity_method` chooses how that is done: local suppression (the default) or
    Mondrian generalization, which writes each group's range (or midpoint, with
    `range_style`) in place of the quasi-identifier. `hierarchies` maps a column to a
    generalization hierarchy that Mondrian uses to describe a group of categories. With
    `enforce_l_diversity`, groups with fewer than `l_target` distinct values of a sensitive
    attribute are then blanked and merged. `t_target`, when set, is the largest
    distance (earth mover's distance, 0 to 1) a group's sensitive values may have from the
    whole table's for the result to count as anonymized.

    `joint_microaggregation` lists sets of numeric columns replaced together, in groups, after
    the column actions run and before k-anonymity is enforced.

    `shuffle_rows` says whether the released rows come back in a fresh random order, with a
    fresh index for pandas. None, the default, shuffles under anonymization and keeps the
    input order and index under pseudonymization. Without a shuffle, an anonymized release
    lines up with the original, and with any other release of the same table, row by row.
    """

    model_config = ConfigDict(extra="forbid")

    fidelity: Fidelity
    # Anonymization is the default because it is the safer reading of a plan that
    # does not say which protection it wants.
    protection: Protection = Protection.ANONYMIZATION
    actions: list[ColumnAction]
    direct_identifiers: list[str] = Field(default_factory=list)
    entity_column: str | None = None
    row_key: str | None = None
    quasi_identifiers: list[str] = Field(default_factory=list)
    sensitive_attributes: list[str] = Field(default_factory=list)
    k_target: int = Field(default=5, ge=1)
    l_target: int = Field(default=2, ge=1)
    enforce_k_anonymity: bool = False
    k_anonymity_method: KAnonymityMethod = KAnonymityMethod.LOCAL_SUPPRESSION
    range_style: RangeStyle = RangeStyle.RANGE
    hierarchies: dict[str, GeneralizationHierarchy] = Field(default_factory=dict)
    enforce_l_diversity: bool = False
    t_target: float | None = Field(default=None, gt=0, le=1)
    joint_microaggregation: list[JointMicroaggregation] = Field(default_factory=list)
    shuffle_rows: bool | None = None

    @model_validator(mode="after")
    def _one_action_per_column(self) -> "AnonymizationPlan":
        seen: set[str] = set()
        for action in self.actions:
            if action.column in seen:
                raise ValueError(f"column {action.column!r} has more than one action")
            seen.add(action.column)
        return self

    @property
    def shuffles_rows(self) -> bool:
        """True when the released rows are put in a fresh random order."""
        if self.shuffle_rows is None:
            return self.protection is Protection.ANONYMIZATION
        return self.shuffle_rows

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
