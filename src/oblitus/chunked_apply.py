"""Apply a plan to a CSV or Parquet file in chunks of rows, for files larger than memory.

Only a technique that treats each row on its own can run on a chunk. Keyed noise, hashes,
encryption, masking and rounding qualify: their result for a row does not depend on the other
rows, so a chunked run releases the same values as an in-memory run. A technique that reads the
whole column (ranks, quantiles, means, distinct values) is refused before any row is read, and
so are the table-level steps (k-anonymity enforcement, l-diversity enforcement and joint
microaggregation).

The whole-table protection checks still run. They read the released file lazily, a few columns
at a time, after the last chunk is written. The value-by-value checks compare the same keyed
sample of at most `MAX_SAMPLE_ROWS` rows that an in-memory run uses.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl

from oblitus.errors import PlanError
from oblitus.evaluation import PrivacyReport, UtilityReport, compare_tables
from oblitus.evaluation.aligned_sample import SampleAccumulator
from oblitus.plan import AnonymizationPlan, Protection
from oblitus.protection import ProtectionAssessment, assess_protection
from oblitus.table import ROW_KEY_COLUMN, row_key_problem
from oblitus.table_files import TableChunkWriter, iter_table_chunks, scan_table
from oblitus.techniques import Technique

DEFAULT_CHUNK_ROWS = 500_000

# Date techniques parse text one distinct value at a time, and the layout they accept is
# guessed from the values of the chunk, so text dates could be read differently per chunk.
_DATE_TECHNIQUES = frozenset({"truncate_date", "date_shift"})

# The original and the released rows of one chunk, each carrying `ROW_KEY_COLUMN`.
ReleasedChunk = tuple[nw.DataFrame[Any], nw.DataFrame[Any]]


@dataclass(frozen=True, slots=True)
class ChunkedResult:
    """What a chunked run wrote and how private and how faithful it is.

    The released table is the file at `destination`. Rows keep the input order, so
    the plan's `shuffle_rows` must be false. The privacy figures cover every row, and the
    utility and value-by-value checks cover the keyed sample.
    """

    destination: Path
    rows: int
    chunks: int
    plan: AnonymizationPlan
    privacy_before: PrivacyReport
    privacy_after: PrivacyReport
    utility: UtilityReport
    protection: ProtectionAssessment


def chunk_blockers(
    plan: AnonymizationPlan, techniques: Mapping[str, Technique], schema: pl.Schema
) -> list[str]:
    """Every reason the plan cannot run in chunks, or an empty list when it can."""
    reasons = [
        f"{action.column}: {action.technique} {reason}"
        for action in plan.actions
        if (reason := techniques[action.technique].chunk_blocker(action.params)) is not None
    ]
    reasons += [
        f"{action.column}: {action.technique} reads text dates with a layout guessed per chunk; "
        "convert the column to a date type first"
        for action in plan.actions
        if action.technique in _DATE_TECHNIQUES and schema[action.column] == pl.String
    ]
    reasons += _table_level_blockers(plan)
    return reasons


def _table_level_blockers(plan: AnonymizationPlan) -> list[str]:
    reasons = []
    if plan.row_key is None:
        reasons.append(
            "the plan needs a row_key, so every row keeps the same noise whatever the chunk"
        )
    if plan.enforce_k_anonymity:
        reasons.append("enforce_k_anonymity needs the whole table")
    if plan.enforce_l_diversity:
        reasons.append("enforce_l_diversity needs the whole table")
    if plan.joint_microaggregation:
        reasons.append("joint_microaggregation needs the whole table")
    if plan.protection is Protection.ANONYMIZATION and plan.shuffles_rows:
        reasons.append(
            "rows keep the input order in a chunked run, so an anonymization plan must set "
            "shuffle_rows to false"
        )
    return reasons


def apply_in_chunks(
    source: Path,
    destination: Path,
    plan: AnonymizationPlan,
    techniques: Mapping[str, Technique],
    *,
    chunk_rows: int,
    release_chunk: Callable[[pl.DataFrame], ReleasedChunk],
    measure_privacy: Callable[[nw.LazyFrame[Any]], PrivacyReport],
    sample_key: bytes,
) -> ChunkedResult:
    """Run the plan on `source` chunk by chunk and write `destination`.

    `release_chunk` turns one chunk into its original and released rows, both carrying
    `ROW_KEY_COLUMN`. `measure_privacy` reads a lazy table, so the whole input and the whole
    output are measured without loading either.
    """
    if source.resolve() == destination.resolve():
        raise PlanError("the output file must differ from the input file")
    original_scan = scan_table(source)
    reasons = chunk_blockers(plan, techniques, original_scan.collect_schema())
    if reasons:
        raise PlanError("this plan cannot run in chunks: " + "; ".join(reasons))
    _require_unique_row_key(original_scan, plan.row_key)

    sample = SampleAccumulator(sample_key)
    rows = 0
    chunks = 0
    with TableChunkWriter(destination) as writer:
        for chunk in iter_table_chunks(source, chunk_rows):
            original, released = release_chunk(chunk)
            sample.add(original, released)
            writer.write(released.drop(ROW_KEY_COLUMN).to_native())
            rows += len(chunk)
            chunks += 1
        written_types = writer.schema

    released_scan = nw.from_native(scan_table(destination, schema=written_types))
    # The streaming engine keeps a pass over a large file within a bounded amount of memory.
    with pl.Config(engine_affinity="streaming"):
        privacy_before = measure_privacy(nw.from_native(original_scan))
        privacy_after = measure_privacy(released_scan)
        aligned = sample.result()
        protection = assess_protection(plan, techniques, privacy_after, aligned, released_scan)
    return ChunkedResult(
        destination=destination,
        rows=rows,
        chunks=chunks,
        plan=plan,
        privacy_before=privacy_before,
        privacy_after=privacy_after,
        utility=compare_tables(aligned.original, aligned.released),
        protection=protection,
    )


def _require_unique_row_key(original_scan: pl.LazyFrame, row_key: str | None) -> None:
    # Each chunk checks its own keys. A key repeated between chunks would share its noise.
    if row_key is None:
        return
    with pl.Config(engine_affinity="streaming"):
        problem = row_key_problem(nw.from_native(original_scan), row_key)
    if problem is not None:
        raise PlanError(problem)
