# Architecture

This document is the map of the code. Read it before your first change.

## Bird's-eye view

A pandas, Polars or DuckDB table comes in, and the main path runs in four steps:
1. The detector classifies each column.
2. The proposal turns the findings into an editable plan, for a protection target
   (pseudonymization or anonymization) and a fidelity (statistical or privacy).
3. The anonymizer runs the plan's techniques column by column, and optionally enforces
   k-anonymity.
4. It measures privacy, statistical drift and the protection level reached.

Beside that path:
- The noise explorer sweeps noise techniques over one column and renders an interactive
  page.
- The synthesis package generates tables from a schema with planted patterns, or learned
  from a reference table, and can wrap a schema's question into a data-science challenge.

Everything is in-process. The only file I/O is in `table_files.py`,
`noise_explorer/page.py`, `cli.py`, `token_vault.py` (the SQLite vault), `table_manifest.py`,
and the JSON `save` and `load` of `plan.py` and `synthesis/schema.py`.

## System views (C4-style)

These three views follow the C4 model's levels, drawn as plain Mermaid flowcharts. They show
the system as it is: a library and a command line that run inside the user's own process,
with no service, database or network call of their own.

### Level 1: system context

Who uses oblitus, and what it touches.

```mermaid
flowchart LR
  owner["Data owner or analyst<br/>(person)"]
  recipient["Data recipient<br/>(person: provider, team, challenge participant)"]
  keyholder["Key holder<br/>(person who may re-link or decrypt)"]
  system["oblitus<br/>(Python library and CLI)"]
  files[("Files<br/>CSV and Parquet")]
  frames[("In-memory and DuckDB tables<br/>pandas, Polars, DuckDB")]
  browser["Web browser<br/>(opens the explorer page)"]

  owner -->|"scans, plans, applies, synthesizes"| system
  system -->|"reads and writes"| files
  system -->|"reads and returns the same kind"| frames
  owner -->|"hands over a released table, or a challenge without answers/"| recipient
  system -->|"explorer page, a local HTML file"| browser
  keyholder -->|"reveal with the secret key"| system
```

- The data owner runs every step and decides what leaves their machine. The library sends
  nothing anywhere.
- The recipient gets only what the owner hands over: a released table, a challenge folder
  without its `answers/`, or a synthetic table.
- The key holder can re-link pseudonyms and decrypt reversible columns. Under anonymization,
  pseudonyms, encryption, codes and fakes use a throwaway key, so nobody can re-link them, but
  the secret key still drives the noise. Its holder can reproduce that noise, so the key stays
  secret.

### Level 2: containers

The runnable parts and the data they produce.

```mermaid
flowchart TB
  subgraph system["oblitus"]
    cli["anonymize CLI<br/>cli.py, argparse"]
    library["Python library<br/>Anonymizer facade and modules"]
  end
  env[("OBLITUS_SECRET_KEY<br/>environment variable")]
  plan[("Plan JSON<br/>editable")]
  outputs[("Table files, report JSON,<br/>challenge folder")]
  page[("Explorer page<br/>self-contained HTML")]
  duckdb[("DuckDB connection<br/>optional, the user's own")]

  cli -->|"calls"| library
  env -->|"read only by the CLI"| cli
  cli -->|"reads tables, writes outputs"| outputs
  library -->|"saves and loads"| plan
  library -->|"save_page writes"| page
  library -->|"runs SQL on the user's relation"| duckdb
```

- The CLI parses arguments, reads the key from the environment, reads and writes the table
  files, reports and challenge folders, and prints results. The library never reads
  environment variables.
- On a DuckDB relation the library builds lazy queries on the user's own connection, so a
  large table stays in DuckDB until the result is collected.

### Level 3: components of the library

```mermaid
flowchart LR
  facade["anonymizer.py<br/>Anonymizer facade"]
  detection["detection/<br/>column categories"]
  proposal["proposal.py and untreated_columns.py<br/>the plan"]
  techniques["techniques/<br/>one expression per column"]
  table["table.py and engines.py<br/>pandas, Polars, DuckDB"]
  suppression["local_suppression.py<br/>k-anonymity"]
  protection["protection.py, identifying_columns.py,<br/>evaluation/: the checks"]
  synthesis["synthesis/<br/>schemas, generation, challenges"]
  explorer["noise_explorer/<br/>sweep and page"]

  facade --> detection
  facade -->|"findings"| proposal
  facade --> techniques --> table
  facade --> suppression
  facade --> protection
  facade --> synthesis
  explorer --> techniques
  synthesis --> detection
```

The code map below names every module and what must not live in it.

## Code map

| Module | What lives here | What must not live here |
|---|---|---|
| `taxonomy.py` | `ColumnKind`, `DataCategory`, `Sensitivity` and the category-to-sensitivity table | Logic that looks at data |
| `detection/` | `checksums.py` (check-digit and structure rules per identifier), `patterns.py` (recognizers: pattern, checksum, strong-check, free-text and name-hint flags), `column_names.py` (name hints, qualifier words and negative hints, EN/PT), `missing_values.py` (placeholder text that counts as missing), `value_evidence.py` (scores every recognizer on a sample and picks the best fit, flagging ties and mixed columns), `person_names.py` (do values look like names), `text_normalization.py` (NFKC, hidden characters, look-alikes and `[at]` undone, with the original position of each character), `text_entities.py` (personal data inside text, found on the normalized copy and mapped back, for redaction and the residue check), `free_text.py` (prose that embeds personal data), `kinds.py` (storage kind of a column and which kinds can hold a category), `detector.py` (`scan`, `ColumnFinding`, fallbacks) | Decisions about what to do with a column |
| `table.py` | `Table`: the table a plan is applied to, on any engine. Techniques read and replace columns through it: statistics, distinct values, keyed uniforms per row or entity, lookups of values computed in Python, reordering values, and each column's original values | Engine-specific code |
| `engines.py` | Everything that differs between pandas, Polars and DuckDB: converting frames in and out, keyed uniforms from SHA-256 (Python or DuckDB SQL), and lookup tables built on the data's own connection | Technique logic |
| `surrogates.py` | `SurrogateFactory`: realistic fake values per category, seeded for consistency | Anything about plans |
| `techniques/` | One module per family (`masking`, `pseudonymization`, `encryption`, `generalization`, `rescaling`, `noise`, `perturbation`). Each technique is `(table, column, context, **params) -> Narwhals expression`, so one implementation runs on every engine. `ff1.py` is the FF1 cipher that `encryption` uses. `numeric_expressions.py` and `date_expressions.py` hold shared expression pieces (Box–Muller and Laplace noise, integer round-trips, day shifts). `tokenization.py` swaps values for random tokens kept in a vault. `value_mapping.py` runs a Python text transform per distinct value or per row and joins the results back. `catalogue.py` registers every technique as a `Technique`, with an `inverse` for reversible ones. `context.py` holds the secret key, the optional token vault, the randomness key with one derived key per column and purpose, HKDF key derivation and fakers | Table-level logic; techniques see one column |
| `token_vault.py` | `TokenVault` (the interface tokenization needs), `MemoryVault`, and `SqliteVault`, a SQLite file created readable by its owner only. The caller passes a vault to `apply` and `reveal`; a plan never holds one | Choosing tokens; techniques decide that |
| `table_manifest.py` | `TableManifest`: the JSON file `apply-many` reads, listing each table's input, output and plan | Applying plans |
| `plan.py` | `Fidelity`, `Protection`, `ColumnAction`, `AnonymizationPlan` (pydantic, JSON round-trip) | Rules for choosing techniques |
| `proposal.py` | `_direct_identifier_rule`, the attribute rule table with its anonymization overrides, and `propose_plan` | Applying techniques |
| `protection.py` | `assess_protection`: the level an applied plan reached (anonymized, at_risk, pseudonymized, identifiable), from the record isolation, linkage and inference checks, run on the released data | Changing data |
| `identifying_columns.py` | Released columns that single out rows or link an entity's rows: fingerprints and untreated entity keys (single or paired columns), counted on the whole table | Changing data |
| `untreated_columns.py` | `treat_untreated_columns`: under anonymization, completes a proposed plan from the whole table's values (fresh codes for text keys, generalized fingerprint timestamps and numbers, low-cardinality columns as quasi-identifiers) | Applying techniques |
| `row_order.py` | `shuffle_rows`: the fresh random row order of an anonymized release | Measuring privacy |
| `local_suppression.py` | `enforce_k_anonymity`: blanks quasi-identifiers in rows of small groups, and the smallest complete group when blank rows stay below k, on the whole table in its engine | Measuring privacy |
| `evaluation/` | `privacy.py` (k-anonymity, l-diversity, on the whole table), `aligned_sample.py` (the same at most 200,000 rows of the original and released table, as pandas, for the checks below), `utility.py` (mean/std change, KS, total variation, correlation change), `disclosure.py` (record linkage, joint linkage over several columns, interval disclosure and per-record shift for a noisy column), `dominance.py` (whether a released number still reveals a segment's dominant contributor) | Changing data |
| `noise_explorer/` | `sweep.py` (runs noise techniques at increasing strengths and measures each), `page.py` and `template.html` (the self-contained interactive page) | Noise algorithms; they come from `techniques/` |
| `synthesis/` | `schema.py` (`DatasetSchema`, columns, overrides, correlations, constraints, `Question`), `expressions.py` (the safe expression language), `marginals.py` (inverse distribution functions), `patterns.py` (conditions, formulas, outliers, exact statistics, gaps), `generator.py` (the four-step build and constraint filtering), `column_values.py` (the values of one column: uniforms, draw or formula, conditions, outliers), `panel.py` (a panel's settings, row layout, column levels and lags of a finished table), `panel_generation.py` (builds a panel period by period), `time_series.py` (the `time_series` section of an answer key), `draw_replay.py` (the uniform numbers behind every row, stored with it so a build can be replayed), `causal_graph.py` (the graph a schema declares, column roles and back-door adjustment sets), `interventions.py` (`do(treatment = value)` by replaying the stored numbers), `treatment_effects.py` (the `causal` section of an answer key), `unique_values.py` (unique integer keys and templates), `copula.py` (Gaussian copula learned from a reference, optionally per level), `latent_correlation.py` (correlation of the hidden normal variables behind categories and numbers), `inference.py` (schema from a table), `rare_values.py` (merges rare categories and pulls in extreme numbers so one record cannot reappear), `challenge.py` (train/test split, answer key and leakage flags), `drift.py` (shifted test rows and drift kind), `label_noise.py` (flipped or perturbed labels) | Detection rules; it calls `detection.scan` |
| `key_generation.py`, `entropy_sources.py` | `generate_key`: mixes `os.urandom` (always) with extra entropy sources through HKDF, health-checks physical samples and reports each source. `entropy_sources.py` holds the built-in readers (hardware RNG, microphone, camera, CPU jitter, beacons) | Anything about plans or techniques |
| `anonymizer.py` | `Anonymizer`, the facade that wires the above together (`scan`, `propose`, `apply`, `apply_many`, `reveal`, `synthesize`, `challenge`), and `AnonymizationResult` | Rules; it only orchestrates |
| `table_files.py` | Read and write CSV or Parquet by extension, with Polars | Anything else |
| `cli.py` | The `oblitus` command; reads `OBLITUS_SECRET_KEY` | Logic beyond parsing arguments and printing |

## Data flow

```mermaid
flowchart LR
  table[DataFrame] --> scan[detection.scan] --> findings[ColumnFinding list]
  findings --> propose[proposal.propose_plan] --> untreated[untreated_columns]
  untreated --> plan[AnonymizationPlan JSON]
  plan --> apply[Anonymizer.apply]
  table --> apply
  apply --> techniques[techniques catalogue] --> suppression[local_suppression]
  suppression --> shuffle[row_order.shuffle_rows] --> result[AnonymizationResult]
  apply --> evaluation[evaluation and protection checks] --> result
  table --> infer[synthesis.infer_schema] --> schema[DatasetSchema]
  schema --> generate[synthesis.generator] --> synthetic[synthetic DataFrame]
  table -. statistical fidelity .-> copula[synthesis.copula] --> generate
```

## Rules that must hold

- A technique receives one column and a `TechniqueContext`, and returns a new column of
  the same length and index. It never mutates its input.
- Missing values stay missing through every technique except `suppress`, which blanks
  every cell, and `shuffle`, which moves them between rows.
- Noise per row comes from keyed hashes of the row key (`context.stream_key`, computed by
  `engines.add_keyed_uniforms`), so the same row gets the same noise on every engine. Random
  choices that stand in for values, such as category codes and random fakes, use
  `context.choice_key` and `context.generator`. All pseudonyms and keyed salts come from
  `context.keyed_digest`, and all encryption keys from `context.derived_key`, so
  `secret_key` reproduces them. Cryptographic nonces, random salts and tokens come from
  `os.urandom`, never from a seeded generator.
- A reversible technique's inverse accepts the same parameters as the technique, so one
  plan drives both `apply` and `reveal`.
- Synthesis never copies a direct identifier from a reference table. Identifier columns
  get fake values or fresh unique numbers.
- Every random number used to build a schema row comes from a uniform number stored with
  the row (`draw_replay.py`), so an intervention can replay the build. Constraint filtering
  drops the numbers together with their rows.
- Schema expressions are parsed with `ast` and evaluated node by node, never with `eval`.
  Random functions are allowed only in formulas, so conditions and constraints give the
  same answer every time they are checked.
- An anonymization plan never runs a technique whose `keeps_link` is true, and it runs
  under a throwaway key. `apply_many` shares one throwaway key across the anonymizing tables
  of a call, and never across calls.
- A vault is passed in at `apply` and `reveal` time and never stored in a plan, so plans stay
  plain JSON.
- The library never reads environment variables or configures logging. The CLI does.

## Cross-cutting concerns

- **Errors**: `errors.py`. `PlanError` for plans, `SchemaError` for schemas,
  `TableFormatError` for file types. All inherit from `AnonymizerError`. The CLI turns them
  into exit code 2, and uses exit code 3 when `apply` misses the plan's protection target.
- **Validation**: plans and schemas are pydantic models validated on load. A plan is also
  checked against the table (columns, techniques, parameter names) before anything runs.
- **Logging**: the package logger has a `NullHandler`. Applications configure logging.
- **Testing**: `tests/` uses a shared `people` fixture (`conftest.py`) with Brazilian
  and generic personal data. Techniques are tested as pure functions, and the facade and
  CLI end to end.

## Where to change things

| To change | Go to |
|---|---|
| What the proposal suggests for a category | `_ATTRIBUTE_RULES` and `_ANONYMIZATION_RULES` in `proposal.py`; direct identifiers in `_direct_identifier_rule` |
| A new kind of personal data | `DataCategory` in `taxonomy.py`; a check in `detection/checksums.py` and a recognizer in `detection/patterns.py`, or a hint in `detection/column_names.py`; a generator in `surrogates.py` if fakes are needed; `docs/detection.md` |
| A new technique | a function in the matching `techniques/<family>.py` and a `Technique` entry in `techniques/catalogue.py` (`inverse=` when reversible, `linkable=True` when its output can be tied back to a person); `docs/techniques.md` |
| A protection check | `protection.py` and `docs/protection.md` |
| A privacy, utility or disclosure metric | `evaluation/` |
| What the noise explorer sweeps or shows | `DEFAULT_SWEEPS` in `noise_explorer/sweep.py`; `noise_explorer/template.html` |
| A distribution, pattern or expression function for synthesis | `synthesis/marginals.py`, `synthesis/patterns.py`, `synthesis/expressions.py`, and `docs/synthesis.md` |
| A CLI command or flag | `cli.py` and the README command table |
