# Pseudonymization and anonymization

Every plan has a **protection** target and a **fidelity**.

| | `statistical` fidelity | `privacy` fidelity |
|---|---|---|
| **`pseudonymization`** | Realistic fakes, consistent per person. Light perturbation. Joins and distributions survive | Keyed hashes for identifiers, generalized quasi-identifiers, k-anonymity enforced |
| **`anonymization`** | Random fakes per cell. Quasi-identifiers generalized only as far as k-anonymity needs. Means kept | Identifiers suppressed. Quasi-identifiers generalized. Sensitive values randomized |

**Pseudonymization** replaces identifiers with stand-ins that the key holder can re-link
or reveal. Under GDPR Article 4(5) and the LGPD, the output is still personal data. Its
value is that one dataset can be joined to another, and an authorized team can get back
to the person.

**Anonymization** removes every link to people:
- A plan that names a linking technique is refused before anything runs. That covers
  keyed hashes, Argon2 and bcrypt, consistent surrogates, scrambling and encryption.
  Argon2 and bcrypt with a random salt are refused too, because anyone can test a
  guessed value against the published hash.
- The plan runs under a throwaway key for everything that could link a release: category
  codes and random fakes are fresh in every run, so two releases cannot be joined on them.
  Noise stays keyed on the secret key and seed. Reissuing the same release gives the same
  noise, so averaging repeated releases gains nothing.
- Quasi-identifiers are generalized, and local suppression then blanks cells until each
  combination is shared by at least `k_target` rows. Rows whose quasi-identifiers are all
  blank still form a group; when they number fewer than `k_target`, the smallest complete
  group is blanked too, so no row has to be dropped.

The result is then checked.

## Default proposals

Every direct identifier follows the row for its protection and fidelity:

| | `statistical` | `privacy` |
|---|---|---|
| pseudonymization | `surrogate` (realistic, consistent); `keyed_hash` for record IDs and identifiers without a fake generator | `keyed_hash` |
| anonymization | `surrogate` with `consistent: false` (random per cell); `scramble_categories` for record IDs, so the rows of one entity stay together without the real ID | `suppress` |

The default target is `anonymization`. Free text with personal data always gets
`redact_entities`. Company names get `scramble_categories` (`ORG-…` codes) in every mode.

`scramble_categories` codes are random per run under anonymization. Under
pseudonymization they are keyed (`keyed: true`), so the same label gets the same code in
every release.

Under anonymization, the proposal also gives every untreated text column with at least 20
distinct values that group rows a `scramble_categories` action with prefix `KEY`. The codes are
fresh for every release and keep the grouping, so a model can still use the column while nobody
can look up their own value. Whole-number columns are kept, because they may be measurements
such as counts or hours whose distribution must survive; when one of them groups rows like a
key, the protection report flags it and the release is `at_risk`. Replace it or list it in the
plan.

Two more steps read the whole table under anonymization:
- A timestamp or number left untreated with a different value in nearly every row (90% or
  more distinct) is generalized, since it would single out rows. Statistical fidelity cuts
  timestamps to the day and rounds numbers to three significant figures, which keeps their
  distribution. Privacy fidelity cuts to the year and uses quintile bands, and adds both
  columns to the quasi-identifiers.
- Every other column with at most 50 distinct values joins the quasi-identifiers, so local
  suppression covers the same columns the protection report uses when it looks for unique
  rows.

Pseudonymization proposals are unchanged. CSV files are read with ISO dates parsed as dates.

The proposal picks an entity column: the company column if there is one, otherwise the
record ID. `date_shift` then moves every date of one entity by the same offset, so the
intervals between that entity's events survive.

The other categories:

| Category | pseudonymization, statistical | pseudonymization, privacy | anonymization, statistical | anonymization, privacy |
|---|---|---|---|---|
| url | `surrogate` | `keyed_hash` | random `surrogate` | `suppress` |
| postal_code | keep 3 characters | keep 2 | keep 3 | keep 2 |
| city | `keep` | `suppress_rare` below 10 | `keep` | `suppress_rare` below 10 |
| date_of_birth | `date_shift` ±30 days, per entity | year only | year only | year only |
| age | `microaggregate` groups of 5 | decades | `microaggregate` groups of 5 | decades |
| gender | `keep` | `keep` | `keep` | `keep` |
| geo_coordinate | Gaussian noise about 1 km | 1 decimal | 2 decimals | 1 decimal |
| income | `rank_swap` within 5% | quintiles | `microaggregate` groups of 5 | quintiles |
| health, special category | `keep` | `randomized_response` p = 0.75 | `keep` | `randomized_response` p = 0.75 |
| nationality | `suppress_rare` below 5 | below 20 | below 5 | below 20 |
| marital status | `keep` | `keep` | `keep` | `keep` |
| business segment | `scramble_categories`, segments under 3 rows merged | under 5 | under 3 | under 5 |
| floor area | quartile bands | quartile bands | quartile bands | quartile bands |
| business revenue, rent | `normalize` robust | `normalize` robust | `normalize` index by entity (rank without one) | `normalize` index by entity (rank without one) |

Any other sensitive categorical column is kept with statistical fidelity and gets
`randomized_response` with privacy fidelity.

k-anonymity is enforced in every mode except pseudonymization with statistical fidelity.
When a technique cannot handle a column's storage kind, such as dates stored as text,
the proposal falls back:
- anonymization: `suppress`;
- pseudonymization: `keyed_hash` for identifiers, and `keep` with "choose a technique by
  hand" for everything else.

## Tokenization vaults

`tokenize` is pseudonymization: the vault maps each token to its value, so anyone who can
read the vault can re-link every record. Treat the vault as you treat the original data:
keep it away from the recipient of the released table, restrict who may read it, and delete it
when no one needs to reveal tokens any more, which cuts the link. The tokens themselves
carry no information about their values. See
[techniques.md](techniques.md#tokenization-and-its-vault). `oblitus apply`, `reveal` and
`apply-many` take `--vault PATH`, a SQLite file.

## The protection checks

`result.protection` holds the outcome:
- `level`;
- `reached`, which is true when the level meets the plan's target;
- `problems`;
- `caveats`.

The checks follow the three criteria of the EDPB Guidelines 02/2026 on anonymisation.

| Criterion | Check |
|---|---|
| No record isolation | Every quasi-identifier combination is shared by at least `k_target` rows (default 5). No row is unique across its discrete released columns, meaning those with at most 50 distinct values, excluding sensitive attributes. No column with at least 20 rows and 90% distinct values is released unchanged, since such a column is a fingerprint. A text column that a technique does not replace (such as `partial_mask`, or `redact_entities` on JSON) counts the same way. An untreated text, integer or whole-number float column (at least 99% of present values whole) outside the plan's lists, with at least 20 distinct values that group rows, is reported too, however many rows each value has: statistics cannot tell an account or card key from a product code, and anyone who knows their own value finds their rows. Pairs of such columns (the 10 with the most distinct values) are checked too: a pair with at least 100 distinct joint values and at most 10 rows on the typical pair is reported as a key that works only together. Replace the column (`scramble_categories` gives fresh codes that keep the grouping) or list it in the plan |
| No linkage | No action uses a linking technique. For noise and perturbation columns, nearest-value matching re-links at most 1 record in `k_target`: no better odds than k-anonymity already accepts. The same limit applies to all perturbed numeric columns taken together, matching each released vector to its nearest original vector on standardized columns. `randomized_response` on a plan with an `entity_column` needs `by`, so a person's rows share one draw |
| No inference | Within each quasi-identifier group, every sensitive attribute takes at least `l_target` distinct values (default 2). A sensitive column released entirely blank is skipped. A numeric sensitive column still in its original units must not let a segment's dominant contributor be read off (dominance and p% rules, see [business-confidentiality.md](business-confidentiality.md)) |

| Level | Meaning |
|---|---|
| `anonymized` | No link to people, and every check passed |
| `at_risk` | No link to people, but a record can still be isolated or a sensitive value inferred. A column cleaned with `redact_entities` always stays here: no pattern finds names written into a sentence or a JSON field |
| `pseudonymized` | Identifiers replaced, but a linking technique lets the key holder re-link them |
| `identifiable` | A direct identifier was left as it was. This is checked on the data: more than 1% of a direct identifier's values surviving unchanged counts, whatever the technique. For `redact_entities`, any text that still holds a recognisable document, email or phone counts |

A column rescaled only by shift and scale (`normalize` with `robust`, `zscore` or `minmax`
and no `by`, whatever `allow_shape_change` says) is a problem under anonymization, not a
caveat: two known values, or one known value plus a true total, undo the scale. Group by the
entity with `allow_shape_change`, or use `rank`.

The `anonymized` level also comes with caveats, because these checks are necessary but
not sufficient:
- columns the detector missed are not treated;
- free text can hide names;
- quasi-identifiers outside the plan, or outside data an attacker holds, are not counted;
- whether the output is anonymous for a given recipient depends on the context. The CJEU
  ruled in *EDPS v SRB* (C-413/23 P, 2025) that pseudonymized data may be anonymous for a
  recipient who has no means to re-identify it.

A contextual assessment, such as the ICO's motivated-intruder test or the ANPD's risk
assessment, is still required.

## Plan files

```json
{
  "fidelity": "privacy",
  "protection": "anonymization",
  "actions": [
    {"column": "email", "technique": "suppress", "params": {}, "reason": "email (direct_identifier): removed"},
    {"column": "age", "technique": "bin_numeric", "params": {"width": 10}, "reason": "age (quasi_identifier): ages in decades"}
  ],
  "direct_identifiers": ["email"],
  "row_key": "customer_id",
  "quasi_identifiers": ["age", "cep", "gender"],
  "sensitive_attributes": ["diagnosis"],
  "k_target": 5,
  "l_target": 2,
  "enforce_k_anonymity": true
}
```

Columns without an action are left unchanged. A plan is checked before anything runs.
A missing column, an unknown technique, an unknown parameter, or a linking technique in
an anonymization plan raises `PlanError`.

`shuffle_rows` says whether the released rows come back in a fresh random order with a fresh
pandas index. `null` (the default) shuffles under anonymization, so the release cannot be lined
up with the original or another release row by row, and keeps order and index under
pseudonymization. The protection checks still compare the rows before the shuffle.

`row_key` names the column that identifies each row, such as a record ID. `propose`
sets it to the first column found to hold a different value in nearly every row. It does
three things:
- Random numbers for noise are keyed on it, so the same row gets the same noise on
  pandas, Polars and DuckDB, and on every run with the same keys.
- A DuckDB table needs it, and its values must be unique, because a DuckDB table has no
  fixed row order. pandas and Polars tables without one use the row position.
- It lines up the original and released rows when utility and the value-by-value protection
  checks compare them. The key holds the original values even when the plan replaces
  the column.

`oblitus apply` and `oblitus reveal` take `--row-key COLUMN`, which overrides the
plan's `row_key`.

## Several tables in one call

`Anonymizer.apply_many(frames, plans, vault=None)` takes related tables and one plan per table,
both keyed by table name, and returns one `AnonymizationResult` per table. One plan per table
keeps each plan an ordinary reviewable file, proposed for its own columns. The tables share
what links a person:

- **Pseudonyms** come from the secret key, so the same value gives the same `keyed_hash` or
  consistent `surrogate` in every table.
- **Date offsets** come from the key and the entity column. List `date_shift` with the same
  `by` column (the person's ID), `max_days` and `stream` in each table's plan, and one person's
  dates move by the same number of days in every table, so intervals between events survive.
  Without `stream`, columns with the same name already agree.
- **Tokens** come from the one `vault`, so a value has one token in every table.
- **Anonymization** runs every anonymizing plan of the call under one throwaway key. Category
  codes and random fakes agree between the tables of this release, and cannot be joined to
  any other release.

Every plan is validated against its table before the first table runs, and a refused plan
names its table. Protection is assessed per table on its own, so a table's level says nothing
about what its columns reveal when joined to another table of the same release. On the command
line, `oblitus apply-many` reads a manifest:

```json
{"tables": {
  "customers": {"input": "customers.csv", "output": "out/customers.csv", "plan": "customers_plan.json"},
  "orders": {"input": "orders.csv", "output": "out/orders.csv"}
}}
```

Relative paths are read from the manifest's folder, and a table without a `plan` gets a
proposed one. The exit code is 3 when any table misses its protection target.

## What keyed noise cannot hide

Noise is computed from the secret key, the seed and the row key, so whoever can recompute it
can subtract it. These limits are not reported per release:

- **A seed without a secret key is public.** Anyone who knows or guesses the seed recomputes
  the noise and gets the original values back exactly. A small seed is found by trying each
  one against a single known record. Releases meant to protect values need a random secret
  key, such as `secrets.token_hex(32)`, kept apart from the release.
- **A guessable key is no key.** A key built from a word, a date or a project name falls to
  the same test: try the candidates against one known value.
- **Re-releasing under the same key shows what changed.** The same row gets the same noise,
  so the difference between two releases of updated data is the exact change in each row.
  Use a new key for each version of the data, or release only once.
- **Releases under different keys or seeds average out.** Each release draws fresh noise, so
  the mean of many releases approaches the true values: the average of 32 releases is about
  six times closer than one. Give each recipient one release.
- **Without a row key, noise follows the row position.** Reordering the input changes which
  row gets which noise, and an attacker who can line up two releases on a public column can
  average them. Set `row_key` when a table is released more than once.
- **DuckDB sees the derived key.** On DuckDB the random numbers are computed in SQL, so the
  per-column key derived from the secret key appears in the query text and in any query log
  or profile. It reveals that column's noise, not the secret key. Do not run releases on a
  connection whose query log others can read.

## What the checks cannot catch

The protection report checks the released table against fixed rules. A release can pass them
and still let someone who knows their own records follow those records:

- **Small keys and combinations.** The report flags an untreated text or whole-number column
  with at least 20 distinct values that group rows, and pairs of such columns. A key with fewer
  values, a key made of three or more columns, a pair of columns that each have fewer than 20
  values, or a decimal key with a few fractional cells passes. List every column in the plan,
  and drop the ones the recipient does not need.
- **Group sizes and own values survive fresh codes.** Replacing an account with a fresh code
  hides the value, not the group. Someone who knows how many rows they have, or a few of their
  own amounts, can find their code and read the rest of their rows. Perturb or generalize the
  numeric columns as well when that matters.
- **Business figures need an entity.** Customer counts per merchant, revenue shares and trends
  are computed from released rows. The dominance check only sees them when the plan names the
  `entity_column`; see [business-confidentiality.md](business-confidentiality.md).

## Large tables

`Anonymizer` takes pandas, Polars and DuckDB tables and returns the kind it was given.
- k-anonymity, l-diversity, local suppression, record isolation and fingerprint columns
  are computed on the whole table by the engine that holds it.
- Utility, disclosure, dominance, surviving identifiers and noisy-value linkage compare each
  released value with its original. They run on the same at most 200,000 rows of both tables,
  picked by a keyed hash of the row key, so the sample is the same on every run and
  engine. Smaller tables are compared whole.
- Dominance sums only the sampled rows of each contributor, so for a table above the limit
  the shares are estimates.
- A DuckDB `TIMESTAMP WITH TIME ZONE` column is read in the connection's time zone, so date
  techniques see local times. Set the connection's `TimeZone` before applying a plan when the
  release must use UTC.
