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

1. **Narwhals at the public edges.** `Anonymizer.scan`, `propose`, `validate`, `apply`
   and `reveal` accept a pandas, Polars or DuckDB frame, and return the same kind.
   PySpark follows the same route.
2. **Techniques are written as Narwhals expressions**, so one implementation runs on
   every backend. Eager in-memory work runs on Polars.
3. **Per-distinct-value techniques compute a mapping table and join it back.** The
   distinct values are collected to Python and mapped there; the mapping goes back to
   the engine as a table and is joined on the value. `lookup_distinct` works this
   way.
4. **Randomness is keyed per row, not drawn in row order.**
   - Every random number comes from SHA-256 of a key and the row key's text. The key is
     derived with HKDF, from the randomness key, for one column and one purpose.
   - Each digest gives four 52-bit uniform numbers, and normal noise uses Box–Muller
     over two of them.
   - A row-order generator gives different results when Spark repartitions. Keyed
     numbers give the same result on every engine and every partitioning.
   - The key keeps the noise secret. Without it, an attacker who knew the row key
     could recompute the noise and subtract it.
   - The digest is never released, only turned into noise. So a key-prefixed hash is
     safe here, and it runs as plain SQL.
   - The derived key appears in the SQL a lazy engine runs. Query logs that record it
     must be protected like the data.
   - The hash comes from Python for pandas and Polars, from `sha256` for DuckDB, and
     will come from `sha2` for Spark.
5. **The row key** is a column the plan names (`row_key`), such as a record ID. When
   there is none, in-memory tables use the row position. Lazy tables need the plan to
   name one, because their row order is not fixed.
6. **Detection reads one column at a time.** `scan` collects each column to pandas. A
   row sample is not enough, because record-ID detection needs the whole column's
   uniqueness. Metrics that compare original and released values run on the same
   sample of at most 200,000 rows, chosen by a keyed hash of the row key. k-anonymity,
   l-diversity, k-anonymity enforcement and the record-isolation checks run on the
   whole table.
7. **Dependencies:** `narwhals`, `polars` and `pyarrow` are required. pandas stays
   required for now, because detection, synthesis and the sampled metrics use it
   internally. DuckDB (`<2` until Narwhals supports 2.0) is an extra. Spark is planned.

## Consequences

- Seeded outputs change once, because random numbers come from keyed row hashes
  instead of a generator stream. The `--seed` option keeps working: it is mixed into
  the randomness key. With a secret key, noise is reproducible only for the key's
  holder. With a seed alone, anyone who knows the seed can recompute it, which suits
  tests and demos only.
- `rank_swap` now sorts by rank plus keyed noise within its window, instead of
  swapping pairs in a loop. The distribution stays exact.
- On 1.07 million rows, `apply` went from 129 s to 20 s, with the same result.
- Whole-column techniques need window formulations. For example, microaggregation
  becomes the mean over groups of `rank // k`, and shuffle sorts by a keyed hash.
- The Gaussian copula in synthesis still needs numpy and SciPy. It generates in
  memory on Polars; generating on Spark, partition by partition, is later work.
- Spark support then needs only the hash adapter and tests. The techniques do not
  change.
