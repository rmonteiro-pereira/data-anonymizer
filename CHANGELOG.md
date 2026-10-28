# Changelog

## 1.1.0

- **Event dates are quasi-identifiers.** A date that is not a birth date (a visit, a purchase,
  an admission) is now counted with the other quasi-identifiers, and the anonymization
  proposal keeps only its year. Before, it was kept to the day and not counted, so city, sex
  and one visit date singled out 20 to 87% of people in a release labelled `anonymized`.
  Pseudonymization proposals keep the date and count it. A column whose dates are all the
  first of a month, the time axis of a monthly table, is a new `reporting_period` category:
  also a quasi-identifier, but kept as it is with statistical fidelity, so a monthly pattern
  survives.
- **Ages are in five-year bands under anonymization** with statistical fidelity, instead of
  means of groups of five. Group means give nearly every group its own age, which beside a
  date left rows unique and got the ages blanked: on the quick start table, the bands and the
  year keep 87% of ages and nearly every date, where month and means blanked 99% of ages.
- **Random numeric codes are record IDs whatever their shape.** Unique whole numbers spread
  evenly within each digit length are codes, which covers codes of mixed lengths, negative
  codes, short codes and codes with a few repeats. 1.0.2 only caught codes of one length, and
  such columns were still rounded and linked for about 60% of people. The spread is tested
  over the range the codes cover, so a sequence that stops short of a power of ten (1 to
  90,000) counts, and a few short placeholder values do not hide the codes. A column named
  after a quantity, or with values on both sides of zero, stays a number.
- The protection guide lists three more limits: consistent pseudonyms follow the value's
  frequency, rounded columns can join two releases, and deterministic `encrypt` leaks length.

A review of the library's statistical claims, measured by simulation, led to these fixes:

- **The synthetic-data check no longer fails fresh data on exact copies by chance.** The 1%
  limit on the excess of copies over the holdout now also needs a one-sided sign test (p ≤
  0.01). On small discrete tables the plain limit failed 27 to 51% of independent draws; it now
  fails about 1 to 4%, and copies of unique training rows are still found.
- **The answer key says what its best score assumes.** A new `best_achievable_assumes` list
  names hidden drivers that a model cannot learn back, drivers with blank cells and a random
  split of a panel. The best score is described as the truth's score on the test set, which a
  model can beat slightly by chance, not as a ceiling.
- **Answer-key roles.** Blanks that follow the target are a leak; a column computed from a
  driver (a noisy proxy of a hidden one) or correlated with a drawn target is
  `correlated_but_not_causal`, not a distractor; an earlier period of a column built from the
  target is listed with `lags_target`.
- **`majority_class_accuracy`** is the share of the more common class. It was one minus the
  positive rate, which is wrong when positives are the majority.
- **Date shifts agree across tables when an ID is an integer in one and a float in another.**
  Keyed draws now hash a whole-number float as its integer, as pandas stores an integer ID as a
  float once one value is missing. Before, only 1% of such rows got the same shift.
- **The utility report** adds `added_blank_share` per column, the cells a release blanked,
  since local suppression blanks rare cells and the filled cells alone can look unchanged. The
  mean change of a date column is now its shift in standard deviations, not a ratio to days
  since 1970.
- **Differentially private integers decode into their own bin.** When the bins did not split
  the integers evenly, up to 10% of decoded values fell in a neighbouring bin.
- `top_bottom_code` and `multiplicative_noise` are labelled `distorts`, not `preserves
  distribution`. Proposal reasons no longer say "shape kept" for an index or rank, or "3
  significant figures kept" where small values round to zero. The linkage problem text gives
  the number of rows it compared.
- The docs now state: randomized response's estimator; suppression's bias on the filled cells;
  that DP accuracy covers the clamped values; noise mass in empty DP bins; the conservative
  zCDP conversion; date shifts break calendar alignment; truncation moves the declared mean;
  Spearman targets are latent for tied columns; infeasible correlation sets shrink; the learned
  copula loses non-monotone relations and dependent blanks; difficulty does not lower the best
  score of a target without noise.

## 1.0.2

- **Sparse numeric codes are record IDs.** A column of unique integers of at least 6 digits,
  95% of one length (account, phone or medical record numbers), is now a record ID, whatever
  its name. Before, only densely numbered keys were: an account number was rounded to 3
  significant figures, and the release said `anonymized` while the rounded number still
  pointed to one row for about 6 in 10 customers of a 400-row table. Epoch timestamps stay
  numeric. Column names ending in `account`, `account_number`, `conta`, `msisdn`, `matricula`
  or `prontuario` are record IDs too.
- **Long integers never stop a CSV read.** Any column that Polars would read as a 128-bit
  integer (more than 19 digits, including values too long for 128 bits and digits of other
  scripts) is read as text, in whole-file, lazy and chunked reads alike. 1.0.1 handled only
  20 to 38 digits, and `apply --chunk-rows` still crashed on them.
- **A listed empty string matches an empty CSV cell** in `dp-synthesize` and `dp-aggregate`.

## 1.0.1

- **A numeric domain no longer echoes a private value in its error.** Text in a column declared
  numeric (for MST, AIM or any release with a `NumericDomain`) raised pandas' own error, which
  quotes the offending value. It now raises `SchemaError` naming only the column, with the
  original error left out of the traceback.
- **The same fix for aggregates and the synthetic-data check.** Text in a summed or averaged
  column (`aggregate_private`, `oblitus dp-aggregate`) and text in a column that is numeric in
  the training table (`check_synthetic`, `oblitus check-synthetic`) raise `SchemaError` without
  the value. Bytes in a category column are compared as text instead of decoded, so a
  malformed byte no longer appears in an error with its position.
- **Numeric bounds must be less than a float's range apart.** Bounds such as `-1e308` and `1e308`
  are refused when declared. Before, a valid record at the upper bound crashed the release,
  so whether it crashed told an observer that the record was there, without spending budget.
- **CSV input for `dp-synthesize` and `dp-aggregate` no longer depends on Polars' type guess.**
  Columns with listed values are read as text and matched as written, so one record such as
  `other` beside `true` and `false` no longer turns a refusal into a release. Listed numbers and
  booleans match their JSON form.
- **A CSV cell that cannot be parsed is refused without the cell.** Polars' error quoted the
  cell and its offset in the file; every command now reports only the file name. Integers of 20
  to 39 digits are read as text instead of crashing the conversion to pandas.
- These refusals carry no trace of the original error, not even as the exception's context.

## 1.0.0

- **AIM synthesis.** `synthesize_private(..., method="aim")` and `oblitus dp-synthesize --method aim`
  build a differentially private synthetic table with AIM, the iterative successor of MST. Each
  round measures the column group the model gets most wrong; a `workload` (`--workload a,b,c`)
  names groups of three or more columns whose joint distribution matters. The mechanism and its
  budget accounting are in oblitus, on the declared domains. Private-PGM fits the model. It
  needs the new `aim` extra (`mbi` and JAX, no PyTorch).
- `PrivateSynthesis.measured_marginals` lists every column group a synthesizer measured, for
  MST and AIM. `oblitus dp-synthesize` prints "column groups measured" instead of "column pairs
  measured".
- A sample of a Spark table (the noise explorer, the sampled checks) now always has the
  requested number of rows. It kept each row with a probability before, and could come up short.
- The roadmap is complete.

## 0.4.0

- **Short labels with a name are person names.** A column of cells such as `Paciente: Ana Silva`
  (a listed name and at most one other word) is classed as a person name, so the default
  proposal replaces it instead of keeping it.
- **Names in prose match in any capitalization.** `ana Silva` and `Ana silva` are found when
  both the first name and the surname are in the bundled lists, as `ana silva` already was.
- **`apply_many` checks every connected sub-join.** A third table holding only some of the
  entities no longer hides a pair of tables that singles a person out. Groups of up to 6 tables
  are checked in every connected subset, larger ones in every connected pair and the full join.
- **Row-count links cover columns that are also in a named link.** A table that copies `account`
  no longer hides the link between `account` and a differently named `patient` column.
- **k-anonymity counts people when the plan has an `entity_column`.** The reported k and the
  rows-below-target figure count distinct entities per combination of every released
  quasi-identifier, joint microaggregation columns included, so a group the extra column
  splits down to two people is `at_risk` (`PrivacyReport.counts_entities` says which unit).
- **The skew check covers text columns of any cardinality.** A diagnosis column with more than
  50 distinct values no longer switches the check off; only numeric columns above 50 values are
  skipped.
- **Anonymization drops free text by default.** No list or pattern finds every name in prose
  (lowercase and foreign names pass them all), so the proposal suppresses free-text columns
  under anonymization. A kept `redact_entities` column stays `at_risk`, and the problem suggests
  dropping it or reviewing its text. Pseudonymization still redacts.
- **`apply_many` joins the whole graph of tables.** Tables linked through a bridge table are
  joined together, and two columns with different names that code the same entities are linked
  when their values occur equally often (`frequency_links.py`). k-anonymity and inference are
  checked over each connected group of tables, per entity.
- **Joint microaggregation counts people when the plan has an `entity_column`.** MDAV groups
  hold at least `group_size` distinct entities, an entity's rows share one vector, and the
  isolated-record and skew checks count entities, so five identical visits of one person are no
  longer a group of five.
- **Joint microaggregation columns count as quasi-identifiers.** The k-anonymity and skew
  checks now group by the aggregated vector too, so five records that share one released vector
  and one diagnosis make the release `at_risk`.
- **Mondrian does not name rare categories.** A category held by fewer than `k_target` rows in
  the table reads `other` inside a released set instead of showing its name.
- **`apply_many` checks the join.** Anonymized tables that release the same column with
  consistent codes are joined and checked for k-anonymity and inference over entities. A join that
  singles out an entity makes every table of it `at_risk`, naming the other table and the link.
- **Joint microaggregation handles missing values.** Records missing a listed column are
  microaggregated among records missing the same ones, or blanked when too few exist, instead
  of keeping their other values. The protection check now looks for isolated records over the
  listed columns.
- **Skewed groups are flagged.** Under anonymization, a quasi-identifier group where one sensitive
  value is at least 90% of the group and at least 30 points above its share in the table makes
  the release `at_risk`, even when l-diversity passes. The limits are the plan fields
  `inference_confidence` and `inference_gain`.
- **Mondrian no longer publishes extreme values.** A group of numbers or dates is described by
  the cuts that bound it (`under 40`, `40 to under 60`, `60 or more`), and the midpoint style
  publishes the group's median, so one outlier does not show in the release.
- **`check_synthetic` catches copies of rare records.** A generator that appends a few rare
  training records verbatim to thousands of fresh rows passed the whole-table tests. The check
  now also counts copies among the 10% most isolated training records, against the same
  count for the isolated holdout records. A copy is a synthetic row within 1% of the record's
  isolation distance on every column (exact or jittered) or on more than half of them (partial
  copy). Repeated ordinary rows no longer relax the rule; chance is measured from real holdout
  rows instead.
  `SyntheticPrivacyReport.isolated_records` and a line of `oblitus check-synthetic` show it.
- **Names and cities in prose are redacted without the `ner` extra.** A first name from the
  bundled IBGE list followed by capitalized words, and a municipality name, are found in free
  text. A column of notes that mentions them gets `redact_entities` in the default plan, and the
  release check reports any that remain. Names separated by several spaces or tabs are found,
  and so are lowercase names (a listed first name followed by a listed surname) and lowercase
  municipalities of several words after "em", "para", "na", "no" or "cidade de". Everyday-noun municipality names and a cell that holds
  only a name or address are left alone.
- **Tagged identifiers and lowercase passport zones are detected.** Identifier patterns no
  longer treat an underscore as part of a word, so `wallet_0x...`, `cpf_5299...` and
  `0x..._backup` are found in text and redacted. Letters or digits touching an identifier
  still stop a match. A passport machine-readable zone is matched and checked in any case.
- **Synthetic-data checks.** `Anonymizer.check_synthetic` and `oblitus check-synthetic SYNTHETIC
  --real TRAIN [--holdout HOLDOUT]` measure the share of exact copies, the distance to the closest
  real record (Gower-like, for mixed columns) and the nearest-neighbour distance ratio. With a
  holdout, they add the DCR share and a distance-based membership-inference attack with a
  bootstrap interval, and give a pass or fail verdict. See `docs/synthesis.md`.
- **Files larger than memory.** `oblitus apply --chunk-rows N` and `Anonymizer.apply_file`
  process a CSV or Parquet file in chunks and write the output incrementally. Keyed per-row
  techniques release exactly what an in-memory run releases. Techniques that read the whole
  column, k-anonymity and l-diversity enforcement, joint microaggregation, a missing `row_key`
  and an unshuffled-order anonymization plan are refused with a `PlanError` listing every
  reason. The protection level comes from the same whole-table checks, run lazily over the
  written file. `Technique` has a new `whole_column_reason` field.

- **Spark tables.** `Anonymizer` takes PySpark DataFrames (extra `spark`) through the same
  engine layer as DuckDB: the work stays lazy as Spark SQL, keyed noise is computed with
  `sha2` and equals the pandas, Polars and DuckDB values, and the plan needs a `row_key`. The
  engine parity tests run on `sqlframe`, and a local PySpark session checks the same values
  in CI; real PySpark tests are skipped without Java.

- **Tokenization with a vault.** The `tokenize` technique replaces values with random tokens
  (optionally keeping length and format, or with a prefix) and keeps the mapping in a vault you
  pass to `apply` and `reveal`. `MemoryVault` and the SQLite-backed `SqliteVault` are built in,
  and `TokenVault` is the interface for your own. The CLI takes `--vault PATH`. It is
  pseudonymization, so anonymization plans refuse it.
- **Several tables in one call.** `Anonymizer.apply_many` and `oblitus apply-many MANIFEST`
  protect related tables together: the same person gets the same pseudonym, token and date
  offset in every table, and anonymizing plans share one throwaway key for the call. Protection
  is assessed per table. `date_shift` takes a `stream` name so date columns of different names
  share per-entity offsets.
- **Key generation from several entropy sources.** `oblitus keygen [--source ...]` and
  `generate_key` mix `os.urandom` (always included) with hardware RNG, microphone, camera, CPU
  timing jitter and, opt-in, public randomness beacons, with a health check and a per-source
  report. New extras `audio` and `camera`. `Anonymizer(throwaway_key_source=...)` draws the
  anonymization throwaway key from the mix.
- **Differential privacy.** `Anonymizer.synthesize_private` and `oblitus dp-synthesize` build a
  synthetic table with MST under a declared (epsilon, delta) or rho budget, and
  `Anonymizer.aggregate_private` and `oblitus dp-aggregate` release noisy counts, sums and means
  per group with OpenDP (new `opendp` extra), with the confidence-interval half-width of every
  value. Column domains and bounds must be declared, never read from the data. Every release
  returns an itemized budget report, and a spend above the declared budget raises
  `PrivacyBudgetError`. See `docs/differential-privacy.md`.
- **Class imbalance.** A question's `positive_share` sets how rare the positive class is by
  shifting the log-odds of its `truth_column`. The answer key gains a `class_balance` section
  with the PR-AUC baseline and the best achievable PR-AUC.
- **Missing not at random.** A column's `missingness` makes its blank cells follow its own
  value (`mnar`) or another column (`mar`). The key records each column's mechanism under
  `missingness`.
- **Related tables.** A schema with a `tables` list generates tables joined by foreign keys,
  with child counts per parent, child columns that read parent columns, and aggregates of
  child rows that a parent's target can depend on. `oblitus challenge` writes every child
  table, `oblitus synthesize` writes a folder of tables, and `Anonymizer.synthesize_tables`
  returns them. Treatments and test shifts are refused for related tables. See
  `examples/customers_orders_challenge.json`.
- **Difficulty levels.** A question's `difficulty` (`easy`, `medium`, `hard`) sets target noise,
  distractor columns, label noise and drift; the key gains a `difficulty` section.
- **Task description.** `QUESTION.md` is now written from the schema's question: what to
  predict, files, score, notes on blank cells and rules, without revealing the answer key. A
  question can name its `metric`.
- **Mondrian k-anonymity.** A plan with `enforce_k_anonymity` can set
  `"k_anonymity_method": "mondrian"` to cut the rows into groups of at least `k_target` and
  replace each quasi-identifier with its group's range, midpoint (`"range_style"`), set of
  categories or hierarchy ancestor. Local suppression stays the default.
- **Generalization hierarchies.** The plan's `hierarchies` field and the new
  `generalize_hierarchy` technique take levels, prefix lengths or the `{"preset": "cep"}`
  Brazilian postal code prefix hierarchy.
- **Joint microaggregation (MDAV).** The plan's `joint_microaggregation` replaces several numeric
  columns together by the mean of groups of at least `group_size` records.
- **t-closeness.** `assess_privacy(..., t_target=...)` and `oblitus.evaluation.t_closeness`
  measure the earth mover's distance of each group from the table. A plan's `t_target` makes a
  distant group a protection problem.
- **l-diversity enforcement.** `enforce_l_diversity` in a plan blanks and merges groups that hold
  fewer than `l_target` sensitive values.
- **Attack-based risk.** `oblitus.evaluation.assess_attack_risks` and `oblitus apply
  --attack-risks` run Anonymeter's singling-out, linkability and inference attacks and report
  each risk with a confidence interval. They need `pip install "oblitus[anonymeter]"`.
- **Names and addresses inside free text.** `redact_entities` takes `entity_model`
  (`"gliner"` or `"presidio"`, from the new `ner` extra). The model loads only when a text is
  first read, and the protection check on the released text uses it too.
- **Names and cities from their values.** The detector knows Brazilian first names and the 5,570
  municipalities (IBGE public data, bundled), so an unnamed column of people or of cities is
  recognized. Unnamed columns of person names are no longer called record IDs.
- **More identifiers.** Passport machine-readable zones (ICAO 9303 check digits) are national
  IDs. Bitcoin (base58check and bech32) and Ethereum (EIP-55) addresses are the new
  `crypto_wallet` category, with fake values for them. The new `identifiers` extra adds
  `python-stdnum` recognizers for ten more national numbers and `phonenumbers` validation of
  international phones; without it the built-in rules decide as before.
- **Calibrated confidence.** `scan` reports a confidence fitted on a labelled benchmark, so a
  finding at 0.9 is right about nine times in ten. Pass `calibrated=False` for the raw score.
  `uv run python -m benchmarks.calibrate_detection` refits it.
- CI installs every extra except `ner`, which pulls in PyTorch.

## 0.3.0

- **Renamed to oblitus.** The package is `oblitus` on PyPI (`pip install oblitus`), the import
  is `import oblitus`, the command is `oblitus`, and the secret-key variable is
  `OBLITUS_SECRET_KEY`. The repository moved to github.com/rmonteiro-pereira/oblitus; the old
  address redirects. Code written for 0.2.0 changes `data_anonymizer` to `oblitus` in its
  imports, `anonymize` to `oblitus` in its commands, and `ANONYMIZER_SECRET_KEY` to
  `OBLITUS_SECRET_KEY`.
- Releases are published to PyPI by a GitHub Actions workflow with trusted publishing.

- Documentation: a guide to writing and opening the noise explorer page (and a warning that it
  holds real values), C4-style system views in `ARCHITECTURE.md`, the project's standards in
  `CONTRIBUTING.md`, and a README quick start that runs on the bundled example schema.
- The utility report computes the KS statistic without the exact p-value, so `apply` no longer
  prints a numpy warning on large tables.

## 0.2.0

- **The quick start reaches `anonymized`.** Under anonymization, `propose` generalizes
  untreated timestamps and numbers that are distinct in nearly every row (day and three
  significant figures; year and quintile bands with `privacy` fidelity), and lists every
  column with at most 50 distinct values as a quasi-identifier. The value-based steps live in
  the new module `untreated_columns.py`.
- Local suppression reaches k when fully blanked rows are too few, by blanking the smallest
  complete group as well.
- CSV files are read with ISO dates parsed as dates.
- Under anonymization, `propose` indexes business revenue and rent per entity
  (`normalize`, `method: index`, `by` the entity) instead of `robust`, or uses `rank` when no
  entity column is found, so the default mall release reaches `anonymized`. `index` rounds to 6
  decimals, and l-diversity skips a sensitive column with a single released value.
- `allow_shape_change` no longer hides an ungrouped `robust`, `zscore` or `minmax` rescale
  from the report, since that flag only permits a shape change. `docs/protection.md` gains "What the checks cannot catch": small or multi-column
  keys, group sizes that survive fresh codes, and business figures without an entity column.
- **Collusion in a small `OTHER` bucket is documented.** `min_entities` protects against one
  member; two of three members can pool their records, so use privacy fidelity (5) when members
  might collude.
- **An affine rescale is a problem under anonymization.** `normalize` with `robust`, `zscore` or
  `minmax` and no `by` makes the release `at_risk`, naming the inversion (two known values, or
  one known value plus a true total) and the fix (`normalize` by entity with
  `allow_shape_change`, or `rank`). Pseudonymization keeps it as a caveat.
- `surrogate` never draws a fake equal to a real value of the column; random fakes could
  occasionally repeat a real email from the input.
- **`OTHER` segments are large enough too.** `scramble_categories` with `min_count` or
  `min_entities` keeps merging the smallest segments into `OTHER` until it holds the minimum
  rows and entities. If all segments together fall short, the column is blanked.

- **A random split refuses lags of hidden copies of the target.** A published column that reads
  a lag of any column depending on the target, directly or through other columns, is refused,
  not only a lag of the target itself.

- **Untreated key columns are replaced, not guessed by cut-off.** Under anonymization,
  `propose` gives every untreated text column with at least 20 distinct values that group rows
  a `scramble_categories` action, so nobody can look up their own value. Whole-number columns
  are kept, since they may be measurements. For any plan, an untreated text or whole-number
  column like this makes the release `at_risk`
  whatever its rows per value (the median bound is gone), and so does a pair of such columns
  that is unique only together. A decimal column counts as discrete at 99% whole numbers.

- **Documented limits.** `docs/protection.md` gains "What keyed noise cannot hide": public
  seeds, guessable keys, same-key re-releases, averaging across keys, positional row keys and
  the derived key in DuckDB SQL. The business guide states that one known value plus a true
  total undoes robust scaling, and `microaggregate` notes its exact-mean subtraction.

- **Synthesis and challenge leaks closed.**
  - A time-split panel challenge blanks columns that read `lag` of the target after the first
    test period, because they published the previous test row's label. The key names them in
    `time_series.withheld_test_columns`.
  - Learned synthesis merges categories seen fewer than 5 times (20 with `privacy` fidelity)
    into `OTHER`, and caps the extreme tail so one record cannot reappear. Privacy quantile
    points need 20 records beyond them.
  - The causal adjustment set searches published non-descendants of the treatment, so a hidden
    cause blocked by a published column no longer gives `null`, and a cause of the treatment
    correlated with the outcome is a confounder.
  - A numeric effect modifier with 10 or fewer distinct values reports one entry per value.
  - `exact` on a row-level panel column is rejected, because it held in every period.

- **Honest protection reports.**
  - Anonymized releases come back in a fresh random row order with a fresh pandas index; the
    plan field `shuffle_rows` overrides this.
  - New protection problems: untreated account or entity keys, perturbed numeric columns that
    link jointly, partly treated text columns that stay distinct, `redact_entities` columns
    (never `anonymized`, since names may remain) and `randomized_response` without `by`.
  - `redact_entities` normalizes text before reading it, so obfuscated emails, CPFs and phones
    are removed, and the report checks residue the same way.
  - `randomized_response` gains `by`; `scramble_categories` gains `min_entities` and `by`, and
    the proposal sets them for business segments.

- **Time series and panels.**
  - A schema can declare a `panel` of entities over periods. Columns take a `level` (`row`,
    `entity` or `period`), and formulas can read `lag(column, k)`, `period` and `entity_index`,
    so AR(1) series, seasonality and regimes (conditions on `period`) can be planted.
  - A question can set `split: "time"` to hold out the last periods. With a `test_shift`, the
    shifted schema generates the test periods as a regime change in the future.
  - The answer key gains `time_series` and, for columns that read the lagged target,
    `features.lags_target`.
  - `rows` is optional for `synthesize`, `challenge` and the CLI `--rows` when the schema has a
    panel. A treatment on a panel is refused with a `SchemaError`.
  - `examples/store_sales_panel.json` plants entity-level store size, a period-level market
    shock, AR(1) sales with seasonality and a regime at period 24.

- **Causal challenges and leakage traps.**
  - A question can name a `treatment`, `treatment_values` and `effect_modifiers`. The answer
    key gains a `causal` section: the true ATE and ATT, the effect on the `truth_column`, the
    effect per modifier level, the naive difference and its bias, the roles of the columns and
    a valid adjustment set.
  - Effects come from `do(treatment = value)` on the generated rows, replayed from the random
    numbers stored with each row.
  - `features.leaks_target` lists published columns computed from the target. They are no
    longer reported as true drivers or distractors.
  - `examples/promo_uplift.json` plants a confounder, mediator, collider, instrument, effect
    modifiers and a leak.
  - Formulas that call `normal` or `poisson`, use `noise_std`, or sit under a condition draw
    from stored uniform numbers by the inverse distribution function. Their distributions are
    unchanged and their seeded values differ.
  - A regression challenge whose target formula reads a text column no longer fails when the
    noise-free truth is computed.
- **Engines.**
  - `scan`, `propose`, `validate`, `apply` and `reveal` accept pandas, Polars and DuckDB
    tables, and return the same kind.
  - Techniques are Narwhals expressions over a `Table`, so one implementation runs on
    every engine.
  - Noise and random choices come from keyed SHA-256 of a row key, so every engine
    releases the same values for the same key.
  - Plans take a `row_key`, and the CLI takes `--row-key`.
  - Files are read and written with Polars.
  - `apply` on 1.07 million rows went from 129 s to 20 s.
  - Seeded outputs differ from earlier versions.
  - `rank_swap` sorts by rank plus keyed noise within its window.
  - Random-salt Argon2 and bcrypt now salt every row.
  - `pyarrow` is required, and the `parquet` extra is gone; DuckDB is the `duckdb` extra.
  - Custom techniques take `(table, column, context, **params)` and return an expression.
- **pandas, Polars and DuckDB tables.**
  - `scan`, `propose`, `validate`, `apply` and `reveal` accept all three and return the same kind.
  - Plans have an optional `row_key`, set by `propose`; `apply` and `reveal` take `--row-key`.
  - k-anonymity, l-diversity and local suppression run on the whole table in its engine.
  - Utility and the value-by-value protection checks compare the same at most 200,000 rows of
    both tables, picked by a keyed hash of the row key.
  - The noise explorer accepts any of them, and files are read and written with Polars.
- **Secure defaults.**
  - `anonymization` is the default protection target.
  - Secret keys must be at least 16 bytes.
  - `laplace_noise` requires `lower` and `upper`.
  - `keyed_hash` digests are at least 16 hex characters.
  - Deterministic Argon2 and bcrypt use a salt that does not depend on the value.
  - `scramble_characters` keeps distinct values distinct.
  - Integer IDs stored as floats get the same pseudonym as before.
  - Under anonymization with statistical fidelity, record IDs are scrambled, not suppressed.
- **Protection checks run on the released data.**
  - A direct identifier is flagged when its values survive.
  - Free text is flagged when it still holds a recognisable identifier.
  - A unique column released unchanged counts as a fingerprint.
  - A sensitive column released entirely blank is no longer scored for l-diversity.
  - Misspelt column names in the plan's lists raise `PlanError`.
- **Business confidentiality.**
  - `scramble_categories`: opaque, shuffled codes, with small segments merged.
  - `normalize`: recentre and rescale while keeping the shape. Shape-changing methods
    need `allow_shape_change`.
  - Dominance and p% checks for figures still in their original units.
  - Proposal rules for companies, segments, floor area, revenue and rent.
  - `date_shift` with `by`, one offset per entity.
  - See `docs/business-confidentiality.md`.
- **Noise explorer.**
  - Columns over 200,000 values are sampled (`--max-rows`).
  - The axis ignores extreme outliers and reports the values beyond it.
  - Zero values are marked on the log axes.
  - Tooltips stay inside the window.
- Pseudonymization and anonymization as explicit protection targets. Anonymization plans
  refuse linking techniques and run under a throwaway key. Every result gets a protection
  level (anonymized, at_risk, pseudonymized, identifiable). The level comes from record
  isolation, linkage and inference checks, and `apply` exits with code 3 when the target
  is missed.
- `surrogate` gains `consistent: false` for random fakes per cell.
- Detection:
  - New identifiers:
    - Brazil: alphanumeric CNPJ, CNH, PIS/PASEP/NIS, CNS, Título de Eleitor, RENAVAM,
      NF-e keys, vehicle plates and PIX keys;
    - international: IMEI, IPv6, MAC, Aadhaar, Spanish DNI/NIE, Portuguese NIF, Dutch
      BSN, German tax ID, UK NINO, US ITIN and Italian codice fiscale.
  - Scored decisions with tie flags, negative name hints, leading-zero recovery, and
    birth dates recognized from their values.
  - Fixes: bare 10–12 digit numbers (timestamps, order numbers) were classed as phones,
    and IMEIs as payment cards.
  - Phone numbers are recognized in North American 3-3-4, compact E.164 and European
    layouts; CPF, CNPJ, IPv4 addresses, versions and dates no longer match.
  - Payment cards must have a length their network issues. IBAN, DNI/NIE, NINO, codice
    fiscale, alphanumeric CNPJ and plates match in lower case.
  - Values: placeholders (blanks, `N/D`, `000.000.000-00`) count as missing; a strong check
    passes at 50% of values with a review-level confidence; a column that mixes CPF and
    CNPJ is detected; a name that agrees wins ties (`renavam` is no longer a NIS); digit
    strings in text columns get their leading zeros back; negative name hints drop every
    recognizer without a strong check, yet keep the category the name spells out.
  - Names: person-name columns in many spellings (`nome_do_cliente`, `nm_paciente`,
    `mother_name`) and person-word columns with name-shaped values; organizations,
    more street hints and `bairro`; special categories (race, religion, politics, union,
    biometric, genetic), nationality, marital status, segment, revenue, rent and floor
    area. Qualifier words (`count`, `type`, `status`, ...) suppress a name hint.
  - A name-only guess is 0.55, so the plan always asks for review, and 0.3 when the storage
    kind cannot hold the category.
  - Integer columns are record IDs only when the values look like keys, and birth dates
    need mostly adult ages. `scan` reads nested object columns, refuses duplicate column
    names with `TableFormatError`, and refuses a `sample_size` below 1.
  - Free text: `::` and `std::vector` no longer count as IPv6, IMEI is not searched in
    prose, a recognizer must beat its chance rate, and the email pattern runs in linear time.
- Noise explorer: an interactive page per column, from `explore-noise`. Disclosure
  metrics: record linkage, interval disclosure, per-record shift, and Wasserstein / IQR.
- Synthesis patterns:
  - distributions (lognormal, gamma, exponential, Poisson, quantile curves);
  - Spearman correlations, conditional overrides and formula columns, with a safe
    expression language that includes date and random functions;
  - outliers, exact statistics and row constraints.
- Synthesis fixes:
  - the expression language rejects huge powers, text repetition and over-long or deeply
    nested expressions, and reports evaluation errors as `ExpressionError`;
  - a discrete distribution (Poisson) keeps its `minimum` value when truncated, and the
    end values of an integer column's bounds are as likely as the rest;
  - constraints are checked before nulls are applied, unique keys are no longer redrawn
    after filtering, and rules with low acceptance (about 1%) succeed; the error states the
    acceptance rate;
  - `unique: true` on a template retries until distinct and fails when the template is too
    small, and schemas reject it on boolean, datetime and `choices` columns;
  - challenge answer keys say "not computable" when the target has bounds, exact statistics,
    conditions or outliers and no `truth_column`, leave out rows with a missing label or
    truth, fail on empty splits, and take a `positive_label` for text classification targets;
  - `infer-schema` describes numeric columns by quantile points instead of a normal mean
    and std, so zero-inflated and heavy-tailed columns keep their mean and sign under
    privacy fidelity;
  - the copula learned from a reference keeps strong associations between booleans and
    categories, by estimating the hidden correlation by maximum likelihood instead of from
    jittered scores;
  - `stratify_by` on `synthesize` and `generate_from_reference` fits one copula per level
    of category columns that have enough rows;
  - conditional overrides are validated after merging with the column: a mean and std
    replace a uniform or quantile base shape, and new choices reset the weights.
- Data-science challenges, from `challenge`: train and test files, plus an answer key
  with the true drivers and the best achievable scores.

## 0.1.0

First release.

- Column detection from names (English and Portuguese) and values, with checksum
  validation for CPF, CNPJ, card numbers and IBAN.
- Editable JSON anonymization plans, proposed under `statistical` or `privacy` fidelity.
- 26 techniques across masking, pseudonymization, encryption, generalization, noise and
  perturbation, including Argon2id and bcrypt hashing, a choice of HMAC algorithm, and
  reversible AES-SIV, AES-GCM and FF1 format-preserving encryption.
- `reveal` (Python and CLI) to decrypt reversible columns with the secret key.
- k-anonymity enforcement by local suppression; k-anonymity, l-diversity and drift reports.
- Synthetic data from a hand-written schema, or from a reference table through a Gaussian
  copula.
- The `anonymize` command line.
