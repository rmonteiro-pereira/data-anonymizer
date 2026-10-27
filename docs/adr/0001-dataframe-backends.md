# 1. Dataframe backends: Narwhals at the edges, Polars inside, keyed per-row randomness

Status: accepted

## Context

The library was written on pandas. On a 1.07 million-row table, `apply` peaked at
1.1 GB of memory, and grouped work ran on one core. Users hold their data in pandas,
Polars, DuckDB and, later, Spark, and should not have to convert it to pandas first.

Every technique already takes one column and returns one column. Moving them to
another engine is a different job for each of three kinds:

| Kind | Techniques | What another engine needs |
|---|---|---|
| Expressions | rounding, binning, truncation, top/bottom coding, `normalize`, `date_shift`, noise | Column expressions, windows and quantiles |
| Per distinct value | keyed hash, FF1, surrogates, Argon2/bcrypt, `scramble_categories`, `encrypt` | Python code per value. There are few distinct values compared with rows |
| Whole column | rank swap, microaggregation, shuffle, k-anonymity suppression | Sorting, ranks and windows over the whole column |

Narwhals 2.27 was tried on a lazy DuckDB frame. Grouped medians, quantiles, ranks,
grouped ranks, ordered shifts, clipping, string slicing and date offsets all work.
Two things do not:
- Python functions per value (`map_batches`) are not available on lazy backends.
- There is no keyed hash or random-number expression.

DuckDB 2.0 also breaks Narwhals joins; DuckDB 1.4 works.

## Decision

1. **Narwhals at the public edges.** `Anonymizer.apply`, `propose`, `scan`, `reveal`
   and the synthesis output accept a pandas, Polars, DuckDB or PySpark frame and return
   the same kind. pandas becomes an optional dependency.
2. **Techniques are written as Narwhals expressions**, so one implementation runs on
   every backend. Eager in-memory work runs on Polars.
3. **Per-distinct-value techniques compute a mapping table and join it back.** The
   distinct values are collected to Python and mapped there; the mapping goes back to
   the engine as a table and is joined on the value. `map_distinct` already works this
   way.
4. **Randomness is keyed per row, not drawn in row order.** Every random number comes
   from HMAC-SHA-256 of the secret key, the column name and a row key, turned into a
   uniform number. Normal noise uses Box–Muller over two such uniforms. A row-order
   generator gives different results when Spark repartitions. Keyed values give the
   same result on every engine and every partitioning. The key keeps the noise secret:
   without it, an attacker who knew the row key could recompute the noise and subtract
   it. A small adapter per backend supplies the hash: Python for pandas and Polars,
   `sha256` for DuckDB, `sha2` for Spark.
5. **The row key** is a column the plan names, such as a record ID. When there is none,
   eager backends use the row position. Lazy backends need the plan to name one,
   because their row order is not stable.
6. **Detection and inference read a sample.** They collect at most a fixed number of
   rows to Polars, as they already only look at a sample.
7. **Dependencies:** `narwhals` and `polars` become required. pandas, DuckDB (`<2`
   until Narwhals supports 2.0) and PySpark are extras.

## Consequences

- Seeded outputs change once, because random numbers come from keyed row hashes
  instead of a generator stream. The `--seed` option keeps working: it is mixed into
  the key.
- Whole-column techniques need window formulations. For example, microaggregation
  becomes the mean over groups of `rank // k`, and shuffle sorts by a keyed hash.
- The Gaussian copula in synthesis still needs numpy and SciPy. It generates in
  memory on Polars; generating on Spark, partition by partition, is later work.
- Spark support then needs only the hash adapter and tests. The techniques do not
  change.
