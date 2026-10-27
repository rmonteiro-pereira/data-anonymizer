# 2. Spark through the engine layer, and chunked files

Status: accepted

## Context

ADR 1 left two scale gaps: Spark tables, and files larger than memory. Java is not available
on every development machine, so real PySpark cannot run in every test environment.

## Decision

1. **Spark is a lazy engine, like DuckDB.** A PySpark DataFrame is wrapped by Narwhals'
   Spark-like support and stays a query until a result is read. The plan needs a `row_key`.
2. **Keyed uniforms use Spark SQL text through `functions.expr`.** `sha2(concat(prefix,
   key text), 256)` replaces DuckDB's `sha256`. Hex digits are parsed with `instr` and
   arithmetic, not `conv`, because `conv` is not available on every Spark-like backend. The
   result equals the Python definition bit for bit, and a test checks it.
3. **Lookups are created on the user's own session** from rows and a type string, so they join
   with the data. Only the types that lookups use are mapped.
4. **`sqlframe` is the test stand-in.** It has the PySpark API on top of DuckDB, and Narwhals
   treats it as Spark-like, so the engine parity tests run on it. It is a development
   dependency. Real PySpark is the `spark` extra. Its tests are skipped without Java and have
   not been run on a cluster.
5. **Chunked files run per-row techniques only.** `Technique.whole_column_reason` marks what
   reads the whole column. Such techniques, the table-level steps and plans without a `row_key`
   are refused before any row is read. No first pass collects statistics: every such
   technique would also need its result checked against the in-memory run, and refusing keeps
   the claim "equal to an in-memory run" true.
6. **Chunks are read with Polars' CSV parser and pyarrow's Parquet reader.** CSV records are
   split by counting quote characters, with the column types inferred once for the whole file.
   This works on every supported Polars version, unlike Polars' newer batch API.
7. **The protection level is computed after the last chunk.** The whole-table checks scan the
   written file lazily, and the sample checks use a bounded keyed sample that equals the
   in-memory one. A chunked run therefore reports the level an in-memory run would.

## Consequences

- Spark column names cannot contain dots or backticks, and Spark timestamps read back in the
  session time zone.
- Narwhals reports a Spark Sunday as weekday 0, so week truncation maps 0 to 7.
- A chunked run keeps the input row order, so anonymization plans must set `shuffle_rows` to
  false explicitly.
- `gaussian_noise`, `normalize` and the rank techniques cannot run on files larger than memory
  until a statistics pass exists.
