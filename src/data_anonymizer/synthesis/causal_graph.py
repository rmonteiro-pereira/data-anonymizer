"""The causal graph a schema declares, and the roles its columns play for one treatment.

Edges:
- A column is caused by the columns its formula reads and its conditions test. These are
  directed edges, and hidden columns are part of the graph.
- A schema correlation joins two columns by an unobserved common cause. It is a bidirected
  edge, kept here as a latent node that points at both columns.

Roles are worked out from the graph alone, so they hold whatever the data look like.
"""

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_anonymizer.synthesis.schema import ColumnSchema, DatasetSchema

_LATENT_PREFIX = "~latent"


@dataclass(frozen=True, slots=True)
class CausalRoles:
    """What each column is, seen from a treatment and an outcome. Lists follow column order."""

    confounders: list[str]
    mediators: list[str]
    colliders: list[str]
    instruments: list[str]


@dataclass(frozen=True, slots=True)
class AdjustmentSet:
    """A set of published columns that blocks every back-door path, or why none exists."""

    columns: list[str] | None
    reason: str


class CausalGraph:
    """Directed edges between columns, plus unobserved common causes between pairs."""

    def __init__(
        self,
        parents: Mapping[str, Collection[str]],
        correlated_pairs: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._order = list(parents)
        self._parents: dict[str, set[str]] = {name: set(found) for name, found in parents.items()}
        self._with_latents: dict[str, set[str]] = {
            name: set(found) for name, found in parents.items()
        }
        for index, (first, second) in enumerate(correlated_pairs):
            latent = f"{_LATENT_PREFIX}{index}"
            self._with_latents[latent] = set()
            self._with_latents[first].add(latent)
            self._with_latents[second].add(latent)

    @classmethod
    def from_columns(
        cls, columns: Iterable["ColumnSchema"], correlated_pairs: Iterable[tuple[str, str]] = ()
    ) -> "CausalGraph":
        """Build the graph from what each column's formula and conditions read.

        Only the schema's columns are nodes: a panel's entity, time and index columns are not.
        A column read through `lag` is an earlier period, so it adds no edge.
        """
        listed = list(columns)
        names = {column.name for column in listed}
        parents = {column.name: column.dependencies() & names for column in listed}
        return cls(parents, correlated_pairs)

    @classmethod
    def of(cls, schema: "DatasetSchema") -> "CausalGraph":
        """Build the graph of a schema, with its correlations as unobserved common causes."""
        pairs = [correlation.columns for correlation in schema.correlations]
        return cls.from_columns(schema.columns, pairs)

    def ancestors(self, name: str) -> set[str]:
        """Columns with a directed path to `name`."""
        return _reachable(self._parents, [name]) - {name}

    def descendants(self, name: str) -> set[str]:
        """Columns that `name` has a directed path to."""
        return self._descendants_avoiding(name, None)

    def _descendants_avoiding(self, name: str, avoided: str | None) -> set[str]:
        children: dict[str, set[str]] = {column: set() for column in self._parents}
        for column, found in self._parents.items():
            if column == avoided:
                continue
            for parent in found:
                children[parent].add(column)
        return _reachable(children, [name]) - {name}

    def roles(self, treatment: str, outcome: str) -> CausalRoles:
        """Classify the other columns as confounders, mediators, colliders or instruments."""
        after_treatment = self.descendants(treatment)
        mediators = after_treatment & self.ancestors(outcome)
        # A collider is reached from the treatment without passing through the outcome. A
        # column that only reads the outcome is a descendant of both, but not a collider.
        colliders = self._descendants_avoiding(treatment, outcome) & self.descendants(outcome)
        others = set(self._order) - {treatment, outcome}
        confounders = {
            name for name in others - after_treatment if self._confounds(name, treatment, outcome)
        }
        instruments = {
            name for name in others - after_treatment if self._instruments(name, treatment, outcome)
        }
        return CausalRoles(
            confounders=self._in_column_order(confounders),
            mediators=self._in_column_order(mediators),
            colliders=self._in_column_order(colliders),
            instruments=self._in_column_order(instruments),
        )

    def adjustment_set(
        self, treatment: str, outcome: str, published: Collection[str]
    ) -> AdjustmentSet:
        """Find published columns that block every back-door path from treatment to outcome.

        A hidden cause does not make the set impossible when a published column blocks its
        paths, so the search tries the published parents of the treatment, then the published
        confounders, their union and every published non-descendant of the treatment. The first
        set that d-separates is pruned to a minimal one.
        """
        confounders = self.roles(treatment, outcome).confounders
        allowed = self._adjustable(treatment, outcome, published)
        for candidate in self._candidate_sets(treatment, confounders, allowed):
            if self._blocks_back_doors(treatment, outcome, candidate):
                columns = self._minimal(treatment, outcome, candidate, confounders)
                if not columns:
                    return AdjustmentSet(
                        [], "no back-door path is open, so no adjustment is needed"
                    )
                return AdjustmentSet(
                    columns, "adjusting for these columns blocks every back-door path"
                )
        hidden = [name for name in confounders if name not in published]
        if hidden:
            return AdjustmentSet(
                None,
                f"the confounders {', '.join(hidden)} are hidden and no published column "
                "blocks their back-door paths",
            )
        return AdjustmentSet(
            None,
            "no set of published columns blocks every back-door path: "
            "an unobserved common cause links the treatment and the outcome",
        )

    def _adjustable(self, treatment: str, outcome: str, published: Collection[str]) -> set[str]:
        """Published columns that may be adjusted for: not the treatment's or outcome's
        effects, because adjusting for an effect biases the estimate."""
        excluded = self.descendants(treatment) | self.descendants(outcome)
        return {name for name in self._order if name in published} - excluded - {treatment, outcome}

    def _candidate_sets(
        self, treatment: str, confounders: list[str], allowed: set[str]
    ) -> list[set[str]]:
        """Adjustment sets to try, from the most targeted to every adjustable column."""
        parents = self._parents[treatment] & allowed
        published_confounders = set(confounders) & allowed
        return [
            parents,
            published_confounders,
            parents | published_confounders,
            allowed,
        ]

    def _minimal(
        self, treatment: str, outcome: str, valid: set[str], confounders: list[str]
    ) -> list[str]:
        """Drop columns one by one while the rest still blocks every back-door path.

        Columns that are not confounders go first, so a confounder is kept when it can be.
        """
        kept = set(valid)
        removal_order = [name for name in reversed(self._order) if name not in confounders]
        removal_order += [name for name in reversed(self._order) if name in confounders]
        for name in removal_order:
            if name in kept and self._blocks_back_doors(treatment, outcome, kept - {name}):
                kept.discard(name)
        return self._in_column_order(kept)

    def _in_column_order(self, names: Collection[str]) -> list[str]:
        return [name for name in self._order if name in names]

    def _confounds(self, name: str, treatment: str, outcome: str) -> bool:
        """True when `name` opens a back-door path: it reaches the outcome without the treatment
        and causes the treatment or shares an unobserved cause with it or its causes, or it
        causes the treatment and shares an unobserved cause with the outcome or its causes."""
        causes_of_treatment = self.ancestors(treatment) | {treatment}
        reaches_outcome = name in _reachable(self._without(treatment), [outcome])
        if reaches_outcome:
            if name in causes_of_treatment:
                return True
            if self._latent_parents(causes_of_treatment) & self._latent_parents({name}):
                return True
        if name not in causes_of_treatment:
            return False
        outcome_side = self.ancestors(outcome) | {outcome}
        return bool(self._latent_parents(outcome_side) & self._latent_parents({name}))

    def _instruments(self, name: str, treatment: str, outcome: str) -> bool:
        if name not in self.ancestors(treatment):
            return False
        reaches_outcome = _reachable(self._without(treatment), [outcome])
        own_causes = _reachable(self._with_latents, [name])
        return not (own_causes & reaches_outcome)

    def _latent_parents(self, names: Collection[str]) -> set[str]:
        return {
            parent
            for name in names
            for parent in self._with_latents[name]
            if parent.startswith(_LATENT_PREFIX)
        }

    def _without(self, removed: str) -> dict[str, set[str]]:
        """The graph with latent nodes and without one column, for paths that avoid it."""
        return {
            name: found - {removed} for name, found in self._with_latents.items() if name != removed
        }

    def _blocks_back_doors(self, treatment: str, outcome: str, given: set[str]) -> bool:
        """True when the given columns d-separate treatment and outcome once the treatment's
        outgoing edges are cut, which leaves only the back-door paths."""
        back_door_graph = {name: found - {treatment} for name, found in self._with_latents.items()}
        return _d_separated(back_door_graph, treatment, outcome, given)


def _reachable(edges: Mapping[str, Collection[str]], start: list[str]) -> set[str]:
    """Return the nodes found by following `edges` from the start nodes, which are included."""
    found: set[str] = set()
    pending = list(start)
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        pending.extend(edges.get(current, ()))
    return found


def _d_separated(
    parents: Mapping[str, Collection[str]], first: str, second: str, given: set[str]
) -> bool:
    """Test d-separation on the moral graph of the ancestors of the three node sets."""
    relevant = _reachable(parents, [first, second, *given])
    neighbours: dict[str, set[str]] = {name: set() for name in relevant}
    for child in relevant:
        child_parents = [parent for parent in parents.get(child, ()) if parent in relevant]
        for parent in child_parents:
            neighbours[child].add(parent)
            neighbours[parent].add(child)
            for other in child_parents:
                if other != parent:
                    neighbours[parent].add(other)
    pending = [first]
    seen = {first}
    while pending:
        current = pending.pop()
        if current == second:
            return False
        for neighbour in neighbours[current] - given - seen:
            seen.add(neighbour)
            pending.append(neighbour)
    return True
