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
    `poisson(rate)`, `uniform(low, high)`. Each call turns a stored uniform number into its
    value by the inverse distribution function (see [Replaying a build](#replaying-a-build)).

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
  - `leaks_target`: published columns computed from the target (see
    [Leakage traps](#leakage-traps));
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

## Leakage traps

A published column that is computed from the target, directly or through other columns, leaks
the answer. Plant one with a formula that reads the target:

```json
{"name": "refund_issued", "kind": "boolean", "formula": "churned and bernoulli(0.7)"}
```

- The answer key lists every such column under `features.leaks_target`.
- A leaking column is not a `true_driver`, a `correlated_but_not_causal` column or a
  `distractor`, because the target causes it and not the other way round.
- A column that reads another leaking column leaks as well, since the dependency runs through
  the graph.
- A model that uses a leaking column scores above `best_achievable` on the test set.

## Causal questions

Name a `treatment` in the question to turn the challenge into a causal one. The key then holds
the true effect of the treatment on the target, which a participant can estimate from the
published rows and be graded against.

```json
"question": {
  "title": "Does the promotion lift next month's revenue?",
  "prompt": "Estimate how much the promotion raises revenue_next_month, on average and for whom.",
  "target": "revenue_next_month",
  "task": "regression",
  "hidden_columns": ["promo_propensity"],
  "treatment": "promo",
  "effect_modifiers": ["segment", "age"]
}
```

[examples/promo_uplift.json](../examples/promo_uplift.json) is a complete schema:
- `promo` is confounded: customers with a high `past_spend` are more likely to get it, and
  `past_spend` also raises revenue;
- `app_minutes` is a mediator, which the promo raises and which raises revenue;
- `satisfaction_score` is a collider, caused by both the promo and revenue;
- `outreach_wave` is an instrument, which shifts the promo and does nothing else;
- the effect differs by `segment` and `age`, the effect modifiers;
- `reward_points_earned` is computed from revenue, so it leaks the target;
- the hidden `promo_propensity` sits between the causes and the promo.

| Question field | Meaning |
|---|---|
| `treatment` | The column whose effect is measured. It must be in the schema, published, not the target, and not caused by the target |
| `treatment_values` | `[control, treated]`. Optional for a boolean column (`false`, `true`) and for an integer column with `minimum` 0 and `maximum` 1. Any other treatment, such as a category with choices `none` and `promo`, needs it |
| `effect_modifiers` | Boolean, category or numeric columns, not caused by the treatment, to report conditional effects by |

**The graph.** It is read from the schema:
- a column is caused by the columns its formula reads and its conditions test;
- hidden columns are in the graph, and are not published;
- a schema `correlations` entry is an unobserved common cause of its two columns.

**The answer key's `causal` section**:

| Field | Meaning |
|---|---|
| `ate` | Mean of Y(treated) − Y(control) over all generated rows, train and test. Y is the target; a classification target is read as 0 or 1 |
| `att` | The same mean over the rows that were treated |
| `effect_on_truth_column` | When the question has a `truth_column`, its `ate` and `att` on that column, such as on the churn probability. It is the effect without the sampling noise of the label |
| `cate` | The mean effect per level of each effect modifier. A numeric modifier is cut into quartiles. Each entry gives `level`, `rows` and `effect` |
| `naive_difference` | Mean target of the treated minus the control among the published training rows |
| `naive_bias` | `naive_difference` − `ate`: what confounding did to the simple comparison |
| `roles` | Lists of `confounders`, `mediators`, `colliders`, `instruments` and `effect_modifiers` |
| `adjustment_set` | A set of published columns that blocks every back-door path, or `null` |
| `adjustment_set_reason` | Why the set is valid, or why none exists, such as a hidden confounder |

The roles follow from the graph:
- **confounders**: common causes of the treatment and the outcome that the treatment does not
  cause. A column joined by a correlation to the treatment or to one of its causes counts when
  it reaches the outcome without passing through the treatment;
- **mediators**: descendants of the treatment that are ancestors of the outcome;
- **colliders**: columns the outcome causes and the treatment reaches without passing through
  the outcome. A column that only reads the outcome is a leak, not a collider;
- **instruments**: causes of the treatment that reach the outcome only through the treatment,
  counting correlations and shared causes.

The `adjustment_set` is the confounders when all of them are published and nothing else
connects the treatment and the outcome. It is `null` when a confounder is hidden, or when a
correlation links the treatment and the outcome directly. Adjusting for a mediator or a
collider biases the estimate, which is what the example is for.

A treatment that is a category needs `treatment_values`, and the column's formula can produce
the choices, such as `where(bernoulli(0.4), 'promo', 'none')`.

## Replaying a build

The effects come from an intervention, `do(treatment = value)`, on the generated rows.
- Every random number used to build a row is stored with it: the uniform number behind each
  drawn column, behind each `bernoulli`, `normal`, `poisson` or `uniform` call in a formula,
  behind `noise_std`, and behind a condition that redraws a formula column.
- Constraint filtering drops the stored numbers of the rows it drops, so they stay aligned.
- An intervention keeps every column the treatment does not cause, sets the treatment, and
  recomputes its descendants in generation order from the stored numbers, conditions included.
  A row keeps its own noise, so the difference between two interventions is the exact effect
  on that row.
- Outliers and blank cells are not applied again, so the effects describe the structural
  model and not the published table. A descendant that is neither drawn nor computed, such as
  a fake name, keeps its value, and `exact` statistics are not restored on recomputed columns.

Schemas that use none of the new fields build the same tables as before for the same seed,
except where a formula calls `normal` or `poisson`, uses `noise_std`, or sits under a
condition. Those keep their distributions but draw from the stored uniform numbers, so their
seeded values differ from earlier versions.

A model that scores close to `best_achievable` has found the planted pattern. A model that
beats it on the test set has overfitted, or found leakage.
