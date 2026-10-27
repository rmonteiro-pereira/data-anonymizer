"""Generalization hierarchies: how a value becomes coarser, one level at a time.

A hierarchy is written in a plan, so a person can review it. Level 0 is the value itself,
each further level is a coarser stand-in (postal code, region, state), and the level one
past the last explicit level is `*`, which hides the value entirely.

Two forms are accepted:
- `levels`: one mapping per level from a value to its stand-in at that level. A value that
  a mapping does not list stays as it is at that level.
- `prefix_lengths`: keep the first characters of the value and star the rest, for codes
  whose leading characters name a region. `digits_only` drops separators first.

The preset `{"preset": "cep"}` is the second form for Brazilian postal codes. It needs no
external data: the first five digits name a delivery area, and the first three, two and one
digits name ever larger regions.
"""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

TOP_SYMBOL = "*"

_CEP_PREFIX_LENGTHS = [5, 3, 2, 1]


class GeneralizationHierarchy(BaseModel):
    """The levels of coarseness a value can take, from specific to general."""

    model_config = ConfigDict(extra="forbid")

    levels: list[dict[str, str]] = Field(default_factory=list)
    prefix_lengths: list[int] = Field(default_factory=list)
    digits_only: bool = False

    @model_validator(mode="before")
    @classmethod
    def _expand_preset(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "preset" not in data:
            return data
        if data["preset"] != "cep" or len(data) > 1:
            raise ValueError("preset must be 'cep' and stand alone")
        return {"prefix_lengths": _CEP_PREFIX_LENGTHS, "digits_only": True}

    @model_validator(mode="after")
    def _exactly_one_form(self) -> "GeneralizationHierarchy":
        if bool(self.levels) == bool(self.prefix_lengths):
            raise ValueError("give either levels or prefix_lengths")
        decreasing = self.prefix_lengths == sorted(set(self.prefix_lengths), reverse=True)
        if not decreasing or any(length < 1 for length in self.prefix_lengths):
            raise ValueError("prefix_lengths must be positive and strictly decreasing")
        return self

    @property
    def top_level(self) -> int:
        """The level at which every value becomes `*`."""
        return max(len(self.levels), len(self.prefix_lengths)) + 1

    def ancestor(self, value: object, level: int) -> str:
        """The stand-in for `value` at `level`; 0 is the value itself."""
        if level < 0:
            raise ValueError("level must not be negative")
        text = str(value)
        if level == 0:
            return text
        if level >= self.top_level:
            return TOP_SYMBOL
        if self.prefix_lengths:
            return self._prefix_stand_in(text, self.prefix_lengths[level - 1])
        return self.levels[level - 1].get(text, text)

    def lowest_common_level(self, values: Iterable[object]) -> int:
        """The first level at which all `values` share one stand-in."""
        distinct = {str(value) for value in values}
        for level in range(self.top_level + 1):
            if len({self.ancestor(value, level) for value in distinct}) <= 1:
                return level
        return self.top_level

    def sort_key(self, value: object) -> tuple[str, ...]:
        """A key that puts values with a common ancestor next to each other."""
        coarse_to_fine = range(self.top_level, -1, -1)
        return tuple(self.ancestor(value, level) for level in coarse_to_fine)

    def _prefix_stand_in(self, text: str, keep: int) -> str:
        if self.digits_only:
            text = "".join(character for character in text if character.isdigit())
        return text[:keep] + TOP_SYMBOL * max(0, len(text) - keep)
