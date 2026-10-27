"""The entry point: scan a table, propose a plan, apply it, or synthesize a new table."""

import hashlib
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data_anonymizer.detection import ColumnFinding, scan, scan_column
from data_anonymizer.engines import (
    Engine,
    column_names,
    engine_of,
    pandas_columns,
    to_working_frame,
)
from data_anonymizer.errors import PlanError, SchemaError
from data_anonymizer.evaluation import (
    PrivacyReport,
    UtilityReport,
    assess_privacy,
    compare_tables,
)
from data_anonymizer.evaluation.aligned_sample import align_sample
from data_anonymizer.local_suppression import enforce_k_anonymity
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from data_anonymizer.proposal import propose_plan
from data_anonymizer.protection import (
    ProtectionAssessment,
    assess_protection,
    linking_actions,
)
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis import (
    Challenge,
    DatasetSchema,
    build_challenge,
    generate_from_reference,
    generate_from_schema,
    infer_schema,
)
from data_anonymizer.synthesis.rare_values import min_records_for
from data_anonymizer.table import ROW_KEY_COLUMN, Table, row_key_problem
from data_anonymizer.techniques import BUILTIN_TECHNIQUES, Technique, TechniqueContext

# Shorter keys can be guessed, and every pseudonym and encryption key derives from it.
MIN_SECRET_KEY_BYTES = 16


@dataclass(frozen=True, slots=True)
class AnonymizationResult:
    """The anonymized table plus how private and how faithful it is.

    `data` is the same kind of table the plan was applied to: pandas, Polars or DuckDB.
    """

    data: Any
    plan: AnonymizationPlan
    suppressed_cells: int
    privacy_before: PrivacyReport
    privacy_after: PrivacyReport
    utility: UtilityReport
    protection: ProtectionAssessment


class Anonymizer:
    """Detects sensitive columns, proposes and applies plans, and synthesizes data.

    `secret_key` keys every pseudonym. Keep it secret and keep it stable: the same
    key gives the same pseudonyms on every run, which lets tables anonymized
    separately still join. Without a key, a random one is drawn, so pseudonyms
    differ between Anonymizer instances. `seed` makes noise and synthesis
    reproducible.
    """

    def __init__(
        self,
        *,
        secret_key: bytes | str | None = None,
        seed: int | None = None,
        locale: str = "en_US",
        extra_techniques: Iterable[Technique] = (),
    ) -> None:
        if isinstance(secret_key, str):
            secret_key = secret_key.encode("utf-8")
        if secret_key is not None and len(secret_key) < MIN_SECRET_KEY_BYTES:
            raise ValueError(
                f"secret_key must be at least {MIN_SECRET_KEY_BYTES} bytes; "
                "generate one with `openssl rand -hex 32`"
            )
        self._secret_key = secret_key if secret_key is not None else secrets.token_bytes(32)
        self._randomness_key = _randomness_key(secret_key, seed)
        self._sample_key = hashlib.sha256(self._randomness_key + b"|evaluation sample").digest()
        self._seed = seed
        self._locale = locale
        self._techniques: dict[str, Technique] = dict(BUILTIN_TECHNIQUES)
        for technique in extra_techniques:
            self._techniques[technique.name] = technique

    @property
    def techniques(self) -> Mapping[str, Technique]:
        """Every technique a plan may name, built-in and extra."""
        return self._techniques

    def scan(self, frame: Any, *, sample_size: int = 1000) -> list[ColumnFinding]:
        """Classify every column of a pandas, Polars or DuckDB table.

        Polars and DuckDB columns are read into pandas one at a time, so a large table
        is never held in pandas whole.
        """
        seed = self._seed or 0
        if engine_of(frame) is Engine.PANDAS:
            return scan(frame, sample_size=sample_size, seed=seed)
        return [
            scan_column(name, values, sample_size=sample_size, seed=seed)
            for name, values in pandas_columns(frame)
        ]

    def propose(
        self,
        frame: Any,
        *,
        fidelity: Fidelity = Fidelity.STATISTICAL,
        protection: Protection = Protection.ANONYMIZATION,
        findings: list[ColumnFinding] | None = None,
    ) -> AnonymizationPlan:
        """Scan the table, unless findings are given, and propose a plan for it."""
        if findings is None:
            findings = self.scan(frame)
        plan = propose_plan(
            findings, fidelity=fidelity, protection=protection, techniques=self._techniques
        )
        # A record-ID column can still repeat or have gaps. Only a column that tells every
        # row apart may key the noise.
        if plan.row_key is not None:
            working, _engine = to_working_frame(frame)
            if row_key_problem(working, plan.row_key) is not None:
                plan = plan.model_copy(update={"row_key": None})
        return plan

    def apply(self, frame: Any, plan: AnonymizationPlan) -> AnonymizationResult:
        """Run a plan on a pandas, Polars or DuckDB table and measure the result.

        The released table comes back as the same kind of table. An anonymization plan
        that names a linking technique is refused before anything runs, and runs under
        a throwaway key so nothing it produces can be re-linked later. A DuckDB table
        needs the plan's `row_key`, because its row order is not fixed.

        Privacy is measured on the whole table by the engine that holds it. Utility and
        the value-by-value protection checks run on at most `MAX_SAMPLE_ROWS` rows,
        the same rows of the original and the released table.
        """
        self.validate(frame, plan)
        engine = engine_of(frame)
        if engine.is_lazy and plan.row_key is None:
            raise PlanError(
                "a DuckDB table needs the plan's row_key: name a column that identifies each row"
            )
        context = self._context(throwaway_key=plan.protection is Protection.ANONYMIZATION)
        table = _table_for(frame, plan)
        original = table.original_frame()
        self._run_actions(table, plan, context)
        released = table.released_frame()

        suppressed_cells = 0
        suppressed_columns: list[str] = []
        if plan.enforce_k_anonymity:
            outcome = enforce_k_anonymity(released, plan.quasi_identifiers, plan.k_target)
            released = outcome.data
            suppressed_cells = outcome.suppressed_cells
            if suppressed_cells:
                suppressed_columns = list(plan.quasi_identifiers)

        privacy_after = self._privacy(released, plan)
        sample = align_sample(original, released, engine, key=self._sample_key)
        released_table = released.drop(ROW_KEY_COLUMN)
        return AnonymizationResult(
            data=table.as_user_frame(released_table, also_changed=suppressed_columns),
            plan=plan,
            suppressed_cells=suppressed_cells,
            privacy_before=self._privacy(original, plan),
            privacy_after=privacy_after,
            utility=compare_tables(sample.original, sample.released),
            protection=assess_protection(
                plan, self._techniques, privacy_after, sample, released_table
            ),
        )

    def reveal(self, frame: Any, plan: AnonymizationPlan) -> Any:
        """Undo the plan's reversible actions on an anonymized table.

        Needs the secret key the table was anonymized with. Columns whose technique is
        one-way are returned unchanged, and recovered values come back as text. The
        result is the same kind of table as the input.
        """
        self.validate(frame, plan)
        context = self._context()
        table = _table_for(frame, plan)
        for action in plan.actions:
            technique = self._techniques[action.technique]
            if not technique.reversible:
                continue
            try:
                technique.reverse(table, action.column, context, action.params)
            except ValueError as error:
                raise PlanError(f"{action.column}: cannot reveal: {error}") from error
        return table.to_native()

    def validate(self, frame: Any, plan: AnonymizationPlan) -> None:
        """Raise PlanError when a plan cannot run on this table."""
        columns = column_names(frame)
        problems = [
            problem
            for action in plan.actions
            if (problem := self._problem_with(columns, action)) is not None
        ]
        problems += _missing_listed_columns(columns, plan)
        if not problems and plan.protection is Protection.ANONYMIZATION:
            problems = linking_actions(plan, self._techniques)
        if problems:
            raise PlanError("; ".join(problems))

    def _run_actions(
        self, table: Table, plan: AnonymizationPlan, context: TechniqueContext
    ) -> None:
        for action in plan.actions:
            technique = self._techniques[action.technique]
            try:
                technique.apply(table, action.column, context, action.params)
            except (TypeError, ValueError) as error:
                raise PlanError(f"{action.column}: {action.technique} failed: {error}") from error

    def _problem_with(self, columns: list[str], action: ColumnAction) -> str | None:
        if action.column not in columns:
            return f"column {action.column!r} is not in the table"
        technique = self._techniques.get(action.technique)
        if technique is None:
            known = ", ".join(sorted(self._techniques))
            return f"{action.column}: unknown technique {action.technique!r}; known: {known}"
        try:
            technique.check_parameters(action.params)
        except TypeError as error:
            return f"{action.column}: {action.technique} parameters: {error}"
        return None

    def synthesize(
        self,
        rows: int | None = None,
        *,
        schema: DatasetSchema | None = None,
        reference: pd.DataFrame | None = None,
        fidelity: Fidelity = Fidelity.STATISTICAL,
        stratify_by: Sequence[str] = (),
    ) -> pd.DataFrame:
        """Generate a new table of `rows` rows.

        From a schema alone, each column follows its description. With a reference
        table, the schema is inferred when not given, and statistical fidelity also
        keeps the correlations between the reference's columns. `stratify_by` names
        category columns whose levels each get their own distributions and correlations.

        A schema with a panel fixes its own row count (entities × periods), so `rows` may be
        left out or must equal it.
        """
        rows = _resolved_rows(rows, schema)
        if rows < 0:
            raise SchemaError("rows must not be negative")
        rng = np.random.default_rng(self._seed)
        surrogates = SurrogateFactory(self._locale)
        if reference is None:
            if stratify_by:
                raise SchemaError("stratify_by needs a reference table")
            if schema is None:
                raise SchemaError("give a schema, a reference table, or both")
            return generate_from_schema(schema, rows, rng=rng, surrogates=surrogates)

        if schema is None:
            min_count = min_records_for(fidelity)
            schema = infer_schema(reference, self.scan(reference), min_category_count=min_count)
        return generate_from_reference(
            reference,
            schema,
            rows,
            fidelity=fidelity,
            rng=rng,
            surrogates=surrogates,
            stratify_by=stratify_by,
        )

    def challenge(self, schema: DatasetSchema, rows: int | None = None) -> Challenge:
        """Generate a data-science challenge from a schema with a `question`.

        Returns the training table, the test features, the test answers and an
        answer key with the true drivers and the best achievable scores. A schema with a
        panel fixes its own row count, so `rows` may be left out.
        """
        rng = np.random.default_rng(self._seed)
        return build_challenge(schema, rows, rng=rng, surrogates=SurrogateFactory(self._locale))

    def _context(self, *, throwaway_key: bool = False) -> TechniqueContext:
        secret_key = secrets.token_bytes(32) if throwaway_key else self._secret_key
        return TechniqueContext(
            secret_key=secret_key,
            surrogates=SurrogateFactory(self._locale),
            randomness_key=self._randomness_key,
        )

    def _privacy(self, frame: Any, plan: AnonymizationPlan) -> PrivacyReport:
        return assess_privacy(
            frame,
            plan.quasi_identifiers,
            plan.sensitive_attributes,
            k_target=plan.k_target,
        )


def _resolved_rows(rows: int | None, schema: DatasetSchema | None) -> int:
    """Return the requested row count, or the schema panel's when none was requested."""
    if schema is not None:
        return schema.row_count(rows)
    if rows is None:
        raise SchemaError("rows is required unless the schema has a panel")
    return rows


def _table_for(frame: Any, plan: AnonymizationPlan) -> Table:
    try:
        return Table.from_native(frame, row_key=plan.row_key)
    except ValueError as error:
        raise PlanError(str(error)) from error


def _randomness_key(secret_key: bytes | None, seed: int | None) -> bytes:
    # Noise drawn under the secret key is reproducible for whoever holds it and
    # unpredictable for everyone else. A seed alone gives reproducible noise that
    # anyone who knows the seed can recompute, which suits tests and demos only.
    # Neither gives fresh noise on every run.
    if secret_key is not None:
        material = secret_key + b"|seed:" + str(seed).encode()
    elif seed is not None:
        material = b"data-anonymizer public seed:" + str(seed).encode()
    else:
        material = secrets.token_bytes(32)
    return hashlib.sha256(material).digest()


def _missing_listed_columns(columns: list[str], plan: AnonymizationPlan) -> list[str]:
    # A misspelt quasi-identifier would silently drop out of every privacy check.
    listed = {
        "direct_identifiers": plan.direct_identifiers,
        "quasi_identifiers": plan.quasi_identifiers,
        "sensitive_attributes": plan.sensitive_attributes,
        "entity_column": [plan.entity_column] if plan.entity_column else [],
        "row_key": [plan.row_key] if plan.row_key else [],
    }
    return [
        f"{field} names {column!r}, which is not in the table"
        for field, named in listed.items()
        for column in named
        if column not in columns
    ]
