# Changelog

## Unreleased

- **Entity keys are found by how they link rows.** An untreated text, integer or whole-number
  float column with at least 50 distinct values and a median of 10 or fewer rows per value now
  makes the release `at_risk`. The old cut-off of 1,000 distinct values let 999 accounts through.

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
