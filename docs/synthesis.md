# Synthetic data

`Anonymizer.synthesize()` and `anonymize synthesize` generate tables two ways:
- **from a schema** you write, with the patterns and statistics you choose;
- **from a reference table**, learning its distributions and correlations.

`Anonymizer.challenge()` and `anonymize challenge` turn a schema with a question into a
data-science exercise with a known answer.

## Columns

```python
from pathlib import Path

from data_anonymizer import Anonymizer, DatasetSchema

schema = DatasetSchema.load(Path("examples/customers_schema.json"))
synthetic = Anonymizer(seed=1, locale="pt_BR").synthesize(10_000, schema=schema)
```

| Kind | Fields |
|---|---|
| `integer`, `float` | A `distribution` and its parameters (below). Left out, the column is normal when `mean` and `std` are given, follows `quantiles` when those are given, and is uniform between `minimum` and `maximum` otherwise. `minimum` and `maximum` truncate any distribution |
| `boolean` | `true_share` |
| `category` | `choices`, optional `weights` |
| `datetime` | `start`, `end` |
| `text` | `category` for realistic fakes (`email`, `person_name`, `cpf`, `cnpj`, `phone`, ...), a `template` where `#` is a digit and `?` an uppercase letter, or `choices` |
| any | `null_fraction`, and `unique` for integer keys, fake-value categories and templates. A unique template with fewer possible values than rows is a `SchemaError`. `unique` is rejected on boolean, datetime and `choices` columns |

| `distribution` | Parameters |
|---|---|
| `normal` | `mean`, `std` |
| `lognormal`, `gamma` | `mean`, `std` of the values themselves (positive mean) |
| `exponential` | `mean`, optional `minimum` as the start |
| `poisson` | `mean` |
| `uniform` | `minimum`, `maximum` |
| `quantiles` | `quantiles`: probability → value, such as `{"0": 1000, "0.5": 3000, "0.9": 9000, "1": 50000}`, joined by straight lines. `minimum` and `maximum` can stand in for the 0 and 1 points |

`minimum` and `maximum` truncate a discrete shape such as `poisson` without losing the
minimum itself. On an `integer` column they are inclusive: a `uniform` 1 to 6 gives every
face the same share, because the continuous draw is widened by half a unit at each end
before it is rounded.

## Planted patterns

Every pattern below is in [examples/churn_challenge.json](../examples/churn_challenge.json).

**Correlations.** These are Spearman rank correlations between drawn columns: numbers,
dates, booleans and categories.
- Categories rank in the order of their `choices`.
- Booleans rank False below True.
- Pairs are joined into one matrix, and an impossible set is adjusted to the nearest
  valid matrix.

```json
"correlations": [{"columns": ["age", "income"], "value": 0.6}]
```

**Conditional rules.** `conditions` override a column's settings where `when` is true.
- Any distribution field can change, and so can `null_fraction`.
- An override is merged with the column and validated as a whole when the schema loads.
- An override that gives `mean` or `std` and no `distribution` replaces a base `uniform` or
  `quantiles` shape with a normal one, so it must give both `mean` and `std`.
- An override that gives `choices` and no `weights` resets the weights to equal shares.
- Overridden rows keep their rank in the distribution, so the rank order of rows survives
  inside the overridden group. The correlation over the whole table is lower than the
  declared value, because the groups sit at different levels: with Spearman 0.6 between
  `age` and `income`, and `income` raised for 40% of rows, each group still shows about 0.6
  but the whole table shows about 0.3. Set the correlation higher, or check it per group.

```json
{"name": "spend", "kind": "float", "mean": 100, "std": 20,
 "conditions": [{"when": "plan == 'pro' and age >= 30", "mean": 300, "std": 40}]}
```

**Formulas.** A `formula` computes a column from others, and `noise_std` adds normal noise
on top. Formulas run after the columns they read, and a cycle is rejected.

```json
{"name": "sales", "kind": "float", "noise_std": 2,
 "formula": "100 + 0.05 * (days(day) - days('2024-01-01')) + 20 * sin(2 * pi * dayofyear(day) / 365)"}
```

**The expression language.** It is Python syntax, limited to:
- arithmetic, comparisons, `and` / `or` / `not`, and `in [...]`;
- column names, numbers and strings;
- the constants `pi` and `e`;
- these functions:
  - maths: `abs`, `sqrt`, `log`, `log1p`, `exp`, `sin`, `cos`, `round`, `floor`, `ceil`,
    `minimum`, `maximum`, `clip`, `where`, `sigmoid`;
  - dates: `year`, `month`, `day`, `hour`, `dayofweek`, `dayofyear`, `days` (days since
    1970);
  - random draws, per row, in formulas only: `bernoulli(p)`, `normal(mean, std)`,
    `poisson(rate)`, `uniform(low, high)`.

Expressions are parsed and evaluated node by node, never with `eval`, so a schema
file from someone else cannot run code. They also cannot use unbounded time or memory:
- an expression is at most 2000 characters long and 100 levels deep;
- powers of plain numbers run in floating point, and one that overflows is an error;
- `*` does not repeat text or lists, and `%` does not format text;
- any arithmetic or type error during evaluation is reported as an `ExpressionError` that
  names the expression.

**Outliers.** `outlier_fraction` pushes that share of rows `outlier_scale` standard
deviations (default 6) into either tail, ignoring the bounds.

**Exact statistics.** With `exact: true`, the generated column hits its targets exactly
rather than on average:
- numbers: exactly `mean` and `std`;
- categories and booleans: exactly the `weights` shares.

Exactness is lost, without an error, when something moves values after the rescale or
makes the target unreachable:
- `minimum` and `maximum` clip the rescaled values, so bounds that do not leave room for
  the target mean and std give a different mean and std;
- an integer column rounds, so its std is off by a small rounding error;
- `outlier_fraction` replaces values afterwards;
- `constraints` drop rows afterwards, and a condition can redraw rows with its own settings.

**Constraints.** `constraints` are row rules such as `end_date >= start_date`. Rows that
break one are dropped and more rows are drawn.
- Constraints are checked before `null_fraction` blanks cells, so a null never removes a
  row and the null share stays as asked.
- Unique keys are never redrawn. A row that repeats a key already kept is dropped as well,
  so formulas, conditions and constraints that read a key stay true.
- Batches are sized from the share of rows that passed so far. Generation stops after about
  200 draws per requested row, at most 20 million, and the error states the acceptance
  rate it measured. A rule that holds for 1% of rows is fine.
- Statistics marked `exact` are exact before this filtering, and approximate after it.

## From a reference table

```python
synthetic = anonymizer.synthesize(5_000, reference=customers, fidelity=Fidelity.STATISTICAL)
```

- **`statistical`**: a Gaussian copula learns each column's distribution and the
  correlations between numeric, date, boolean and category columns. Numbers are
  interpolated between observed quantiles, so rows are new rather than copied.
  Identifiers are always regenerated as fakes, never sampled from the reference.
  - The correlation behind a boolean or category column is estimated by maximum
    likelihood (polyserial against a number, polychoric against another category), so a
    strong association between two yes/no columns survives: on a reference with phi 0.79,
    the synthetic table shows 0.79.
  - A category is one ordered scale, ranked by how common each value is. A nominal
    category whose effect is not monotone along that scale, such as group means `a` 0,
    `b` 10, `c` 1, is blurred. Fit one copula per level with `stratify_by` for that.
- **`privacy`**: every column is drawn independently from an inferred schema, so no
  row-level structure survives. Categories seen fewer than 5 times become `OTHER`.
  Numeric columns are inferred as `quantiles`: the deciles, the 5% and 95% points, finer
  points in each tail while the sample is large enough, and the minimum and maximum. They
  follow zero-inflated and heavy-tailed shapes, such as a capital gain that is zero for 92%
  of people, where a normal distribution with the same mean and std would invent negative
  values and a wrong mean. The mean of a heavy tail still varies by some percent.

`stratify_by` names category or boolean columns of the reference, such as `["sex"]`:

```python
synthetic = anonymizer.synthesize(5_000, reference=customers, stratify_by=["segment"])
```

- Each level with at least 200 reference rows gets its own copula, so it keeps its own
  distributions and correlations, and appears at its share of the reference.
- Smaller levels, and rows with a missing level, share one pooled copula.
- It needs a reference table and has no effect with `privacy` fidelity. A name that is not a
  category or boolean column learned from the reference is a `SchemaError`.
- Only the levels are separated. Associations between a stratifying column and the rest
  come from the levels, not from the copula.

Patterns in a schema you pass along, such as formulas, conditions and constraints, apply
in both modes. `anonymize infer-schema` writes the inferred schema so you can edit it and
add patterns.

## Data-science challenges

Add a `question` to a schema:

```json
"question": {
  "title": "Who is about to churn?",
  "prompt": "Predict which customers churn in the next month. Score: ROC AUC.",
  "target": "churned",
  "task": "classification",
  "hidden_columns": ["churn_probability"],
  "truth_column": "churn_probability"
}
```

```bash
uv run anonymize --seed 7 challenge --schema examples/churn_challenge.json --rows 20000 -o churn
```

The command writes:

| File | Contents |
|---|---|
| `churn/QUESTION.md` | The question, the task, the target and the column list |
| `churn/train.csv` | Training rows with the target, without the hidden columns |
| `churn/test.csv` | Test rows without the target |
| `churn/answers/test_answers.csv` | The test targets |
| `churn/answers/answer_key.json` | The answer key |

Keep `churn/answers/` away from participants.

The answer key holds:
- **features**. Every column falls into one group:
  - `true_drivers`: published columns the target is computed from, directly or through
    other columns;
  - `hidden_drivers`: drivers that are withheld;
  - `correlated_but_not_causal`: columns correlated with a driver but not used to compute
    the target;
  - `distractors`: everything else.
- **how_the_target_was_built**: every formula, distribution and condition on the path to
  the target.
- **best_achievable**: the best score any model can reach, measured against the noise-free
  truth on the test rows.
  - Regression: RMSE, MAE and R². The truth is the `truth_column`, or the target's formula
    evaluated again without its noise. Without a `truth_column`, the key says "not
    computable" when the target has `minimum` or `maximum`, `exact`, `conditions`,
    `outlier_fraction` or integer rounding, because those change the target after its
    formula and the formula is then not the best prediction.
  - Classification: accuracy, ROC AUC and log loss of the true probability in
    `truth_column`, plus the positive rate. The target is boolean or 0/1. For any other
    target, such as the choices `no` and `yes`, set `positive_label` in the question.
  - Rows with a missing target or truth are left out of the scores.
  - The table needs at least one training row and one test row, or generation fails with a
    `SchemaError`.

A model that scores close to `best_achievable` has found the planted pattern. A model that
beats it on the test set has overfitted, or found leakage.

### Drift between train and test

`test_shift` changes columns in the test rows only. The train rows use the base schema. The
test rows are generated again from the schema with each shift merged into the named column,
and the merged schema is validated like any other. The test size stays `test_share` of `rows`.

```json
"question": {
  "target": "churned",
  "task": "classification",
  "test_shift": [
    {"name": "age", "mean": 30, "std": 8},
    {"name": "churn_probability", "formula": "sigmoid(-1.2 + 1.1 * support_tickets - 0.04 * tenure_months)"}
  ]
}
```

A shift takes the settings of a `conditions` override (distribution, `mean`, `std`, `choices`,
`weights`, `null_fraction`, ...), plus `formula`, `noise_std` and `conditions`, which replace
the column's own. It has no `when`: it applies to the whole test set. Put a `when` in its
`conditions` to shift some rows. Unique keys are unique within each split, not across them.

The key gains a `drift` section:
- `shifted_columns` and `shifts`: what changed;
- `kind`: `concept` when the target is shifted, or a column it is computed from gets a new
  formula, `noise_std` or `conditions`; `covariate` when only distributions change;
- `best_achievable`: the score of the shifted truth on the shifted test rows. It is also the
  top-level `best_achievable`.

A model fitted on the train rows scores worse than `best_achievable` under either kind, and
under a concept shift no amount of training data closes the gap. Examples:
[churn_drift.json](../examples/churn_drift.json) (younger test customers with more tickets)
and [churn_concept_drift.json](../examples/churn_concept_drift.json) (the ticket effect
doubles).

### Label noise

`label_noise` corrupts the published labels after the truth is known and the rows are split.

```json
"label_noise": {"flip_rate": 0.1, "split": "train"}
```

- Classification takes `flip_rate` (above 0 and up to 0.5): each present label flips with that
  probability. The target is boolean, 0/1 or a category with exactly two choices.
- Regression takes `std`: normal noise is added to each present label. An integer target stays
  integer.
- `split` is `train` (the default), `test` or `both`. Test answers are clean unless it includes
  `test`.

The key gains a `label_noise` section with the settings, the number of labels `flipped` (or
`perturbed`) per split, and `best_achievable_on_clean_test_labels`. The top-level
`best_achievable` is measured against the published test answers, so it equals the clean score
unless the test split is noisy.
