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
| any | `null_fraction`, `unique` (fresh unique keys or fakes) |

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
- Overridden rows keep their rank in the distribution, so correlations survive.

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
break one are dropped and redrawn, and unique keys are drawn again over the final rows.
Statistics marked `exact` are exact before this filtering, and approximate after it.

## From a reference table

```python
synthetic = anonymizer.synthesize(5_000, reference=customers, fidelity=Fidelity.STATISTICAL)
```

- **`statistical`**: a Gaussian copula learns each column's distribution and the
  correlations between numeric, date, boolean and category columns. Numbers are
  interpolated between observed quantiles, so rows are new rather than copied.
  Identifiers are always regenerated as fakes, never sampled from the reference.
- **`privacy`**: every column is drawn independently from an inferred schema, so no
  row-level structure survives. Categories seen fewer than 5 times become `OTHER`.

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
    evaluated again without its noise.
  - Classification: accuracy, ROC AUC and log loss of the true probability in
    `truth_column`, plus the positive rate.

A model that scores close to `best_achievable` has found the planted pattern. A model that
beats it on the test set has overfitted, or found leakage.
