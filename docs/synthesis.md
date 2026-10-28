# Synthetic data

`Anonymizer.synthesize()` and `oblitus synthesize` generate tables two ways:
- **from a schema** you write, with the patterns and statistics you choose;
- **from a reference table**, learning its distributions and correlations.

`Anonymizer.challenge()` and `oblitus challenge` turn a schema with a question into a
data-science exercise with a known answer.

## Columns

```python
from pathlib import Path

from oblitus import Anonymizer, DatasetSchema

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
| any | `level` in a schema with a `panel` (see [Panels and time series](#panels-and-time-series)), `null_fraction`, `missingness` (see [Planted patterns](#planted-patterns)), and `unique` for integer keys, fake-value categories and templates. A unique template with fewer possible values than rows is a `SchemaError`. `unique` is rejected on boolean, datetime and `choices` columns |

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

Truncation keeps the shape but moves the declared `mean` and `std`, which describe the
distribution before it is cut: a `normal` with mean 50 and std 10 cut at a minimum of 45 comes
out with mean 55.1 and std 7.0. Use `exact` when the published mean and std must be the declared
ones. A formula column with bounds is clipped instead, so values pile up at the bound.

## Planted patterns

Every pattern below is in [examples/churn_challenge.json](../examples/churn_challenge.json).

**Correlations.** These are Spearman rank correlations between drawn columns: numbers,
dates, booleans and categories.
- Categories rank in the order of their `choices`.
- Booleans rank False below True.
- The value is reached for continuous columns. For columns with many ties (booleans, counts
  with a small mean, categories with few levels) the declared value is the correlation of the
  underlying draws, and the published one is smaller: 0.5 between two fair booleans comes out
  near 0.35, and 0.3 between a 5% boolean and a number near 0.15.
- Pairs are joined into one matrix. An impossible set, such as 0.9, 0.9 and -0.9 among three
  columns, is made valid by raising its negative eigenvalues, which can shrink every
  correlation of the set: that example comes out near 0.48 each, without a warning. Check
  that the declared correlations can hold together.

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

**Missing-value mechanisms.** A plain `null_fraction` blanks cells completely at random
(MCAR). A `missingness` setting makes the chance of a blank cell depend on a value, while
`null_fraction` stays the overall share of blank cells:

```json
{
  "name": "income", "kind": "float", "mean": 4000, "std": 1500,
  "null_fraction": 0.2,
  "missingness": {"mechanism": "mnar", "strength": 1.5}
}
```

- `mnar` (missing not at random): the chance follows the column's own value, as when high
  incomes go unreported. `depends_on` must be left out.
- `mar` (missing at random): the chance follows another column, named in `depends_on`, such
  as older customers skipping a field.
- `strength` is the change in the log-odds of a blank cell per standard deviation of the
  driving value; positive blanks large values more often, negative blanks small ones.
  The curve is shifted so the overall share still equals `null_fraction`.
- The driving column must be a number, a boolean or a date. It is read before any cell is
  blanked, so a driver's own blanks do not matter.
- `null_fraction` must be above 0, and a column cannot also set `null_fraction` in
  `conditions`.

A challenge key lists every column with blank cells under `missingness`: its `mechanism`
(`mcar`, `mar` or `mnar`), what it `depends_on`, its `strength` and the observed share of blank
cells per split. A column whose `conditions` change `null_fraction` is reported as `mar`, with
those conditions.

## From a reference table

```python
synthetic = anonymizer.synthesize(5_000, reference=customers, fidelity=Fidelity.STATISTICAL)
```

- **`statistical`**: a Gaussian copula learns each column's distribution and the
  correlations between numeric, date, boolean and category columns. Numbers are
  interpolated between observed quantiles, so rows are new rather than copied. A category
  seen fewer than 5 times is merged into `OTHER` (or into the most common category when the
  rare ones together are still fewer than 5 rows), and the 5 largest and 5 smallest numbers
  are pulled in to the nearest value beyond them, so a single record cannot reappear.
  Identifiers are always regenerated as fakes, never sampled from the reference.
  - The correlation behind a boolean or category column is estimated by maximum
    likelihood (polyserial against a number, polychoric against another category), so a
    strong association between two yes/no columns survives: on a reference with phi 0.79,
    the synthetic table shows 0.79.
  - A category is one ordered scale, ranked by how common each value is. A nominal
    category whose effect is not monotone along that scale, such as group means `a` 0,
    `b` 10, `c` 1, is blurred. Fit one copula per level with `stratify_by` for that.
  - The copula keeps monotone relations only. A relation such as `y = x²` (correlation 1.0
    between `x²` and `y` in the reference) and an interaction such as `y = a XOR b` vanish in
    the synthetic table.
  - Blank cells keep their share per column but fall at random: blanks that depended on
    another column (more of them among the young, say) lose that dependence.
- **`privacy`**: every column is drawn independently from an inferred schema, so no
  row-level structure survives. Categories seen fewer than 20 times become `OTHER`.
  Numeric columns are inferred as `quantiles`: the deciles, the 5% and 95% points, finer
  points in each tail while at least 20 records lie beyond each point, and a minimum and
  maximum that are the values with 20 records beyond them, so one extreme record cannot shape
  the tail. They
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
in both modes. `oblitus infer-schema` writes the inferred schema so you can edit it and
add patterns.

## With a differential-privacy guarantee

`Anonymizer.synthesize_private()` and `oblitus dp-synthesize` learn a reference table under a total
ε (and δ) budget with MST, or with AIM (`method="aim"`, the `aim` extra), and return the
synthetic table, the budget spent and the column groups the model measured. It differs from the two fidelities above in what it promises: the release is
differentially private, so it protects every record against any attacker, and the cost is
accuracy. The domain of every column (its values, or its bounds and bins) must be declared, not
read from the data. See [differential-privacy.md](differential-privacy.md).

## Checking a synthetic table for memorization

A generator can copy the records it learned from. `Anonymizer.check_synthetic(synthetic, train,
holdout)` and `oblitus check-synthetic SYNTHETIC --real TRAIN [--holdout HOLDOUT]` measure that.
`train` is the table the generator saw. `holdout` is real data from the same population that it
never saw; give one to get the DCR share and the membership attack.

Distances are Gower-like: the mean over the shared columns of a per-column distance between 0 and
1. A category column counts 0 for equal values and 1 for different ones. A number or date counts
its gap in robust standard deviations of the training table (interquartile range over 1.349),
divided by 3 and capped at 1. A blank equals only a blank. Each table is cut to at most 10,000
random rows (`max_rows`), and train and holdout are cut to the same size, because a larger table
is closer to everything. The distances are computed in chunks, so memory stays small, but time
grows with the product of the row counts, which is why there is a cap.

| Measurement | Meaning |
|---|---|
| Exact copies | Share of synthetic rows whose distance to a training row is 0 |
| Distance to closest record (DCR) | Median over synthetic rows of the distance to the nearest training row (and holdout row) |
| NNDR | Nearest over second-nearest distance; near 0 means a row sits on top of one training record |
| DCR share | Share of synthetic rows closer to the training table than to the holdout; 50% means no memorization, ties count half |
| Isolated-record copies | Among the 10% of records with the largest distance to the nearest other record of their own table (and not 0), how many have a synthetic row within 1% of that distance, column by column, on every column (copy) or on more than half of them (partial copy); counted for training and for holdout records |
| Membership attack | The attacker holds the training rows and the holdout rows, guesses "member" for those closest to the synthetic data. AUC is 0.5 when it learns nothing; advantage is 2 * AUC - 1 |

The verdict fails on any of these findings, and passes when there are none:

- **Exact copies**: the share of synthetic rows equal to a training row exceeds the share equal
  to a holdout row by more than 1%, and a one-sided sign test over the rows that equal one
  table but not the other says the excess is not chance (p ≤ 0.01). Rows that repeat in a
  low-cardinality population match the holdout too, so they are not counted; the test keeps
  small discrete tables from failing on chance alone, which the plain 1% limit did in about
  half of 100-row trials. Without a holdout, the limit applies to the share itself.
- **DCR share**: above 60%, and its 95% bootstrap interval excludes 50%.
- **Membership attack**: the 95% bootstrap interval of the advantage lies entirely above 0 and
  the advantage itself is at least 0.1. On fresh draws of about 1,000 rows or more, leak-free
  generators stay below 0.07, so the floor keeps them from failing. Small tables are noisier:
  at 100 rows, 12% of fresh draws reached 0.1, and about 2% raised a membership finding.

- **Isolated-record copies**: the whole-table tests average over every record, so a generator
  that appends five rare training records to a thousand fresh rows passes them (0.5% exact
  copies, membership AUC near 0.51) while anyone who knows one of those people can tell they
  were in the data. A record is isolated when its distance to the nearest other record of its
  own table is among the largest 10% and not 0, so a record with a twin is never isolated and
  repeated rows elsewhere change nothing. A synthetic row copies an isolated record when it
  lies within 1% of the record's own isolation distance on every column (an exact or jittered
  copy), or on more than half of the columns (a partial copy, such as the measurements without
  one value). Fresh draws almost never come that close to a record that is far from all its
  neighbours. The same count is made for equally isolated holdout records, whose isolation is
  measured inside the holdout. Without a holdout, any copy is a finding. With one, any excess
  of training over holdout copies is a finding unless real holdout rows also match training
  records by this rule (categories, rounded or clipped numbers); chance matches then exist,
  and the excess must pass a one-sided sign test at the 1% level, which needs about seven
  copies against none.
  In 40 trials of each case, fresh draws, the copula synthesizer and MST at ε = 1 raised a
  finding from this test in 0 of 360 runs with 1,500 training rows and 5 of 360 with 300 rows
  (fresh draws, discrete table: 1; copula, mixed table: 2, discrete table: 2; MST: 0).
  Appending five rare records to 1,000 fresh rows was caught in 40 of 40 trials when copied
  verbatim, when jittered by 0.0001, 0.001 or 0.01 robust standard deviations, when the
  records were copied on three of four columns, and when ordinary rows repeat in the real
  tables. Remaining limits: a jitter of about 0.03 standard deviations is caught in about
  three trials of four and 0.1 in almost none, because the tolerance is 1% of the isolation
  distance; copying fewer than half of the columns is not caught; a copied record that is not
  among the most isolated 10% is not looked at; and at about 300 rows a single copy cannot be
  told from chance on discrete or mixed data.

Change the limits with `SyntheticPrivacyThresholds`. The command prints the measurements, writes
them as JSON with `--report`, and exits 3 when there is a finding. This is a screening, not a guarantee: the bootstrap covers the sampling of the candidate
records, not the randomness of the generator, which is why the attack also needs the minimum
advantage. Only a differential-privacy mechanism bounds what an attacker can learn.

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
uv run oblitus --seed 7 challenge --schema examples/churn_challenge.json --rows 20000 -o churn
```

The command writes:

| File | Contents |
|---|---|
| `churn/QUESTION.md` | The task description for participants (see [Task description](#task-description)) |
| `churn/train.csv` | Training rows with the target, without the hidden columns |
| `churn/test.csv` | Test rows without the target |
| `churn/<table>.csv` | Related schemas only: every child table, in full (see [Related tables](#related-tables)) |
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
  - `lags_target`: published columns that read an earlier period of the target in a panel (see
    [Panels and time series](#panels-and-time-series)); only present when there are any;
  - `distractors`: everything else.
- **how_the_target_was_built**: every formula, distribution and condition on the path to
  the target.
- **best_achievable**: the score of the noise-free truth on the test rows: what a model that
  had learned the data-generating rule exactly would score on this test set. It is a target,
  not a hard ceiling. On a finite test set a model can beat it slightly by chance (by up to
  about 0.007 in accuracy in 15 trials); beating it by a clear margin points at a leak.
- **best_achievable_assumes**: present when the published files lack something the truth
  uses, so no model trained on them can reach the score: a hidden driver that was drawn or
  carries noise (a hidden driver computed without noise from published columns, such as a
  probability, can be learned back and is not listed), drivers with blank cells, or a random
  split of a panel, where the labels of the periods around a test row sit in training.
  - Regression: RMSE, MAE and R². The truth is the `truth_column`, or the target's formula
    evaluated again without its noise. Without a `truth_column`, the key says "not
    computable" when the target has `minimum` or `maximum`, `exact`, `conditions`,
    `outlier_fraction` or integer rounding, because those change the target after its
    formula and the formula is then not the best prediction.
  - Classification: accuracy, ROC AUC and log loss of the true probability in
    `truth_column`, plus the positive rate. The target is boolean or 0/1. For any other
    target, such as the choices `no` and `yes`, set `positive_label` in the question.
  - Rows with a missing target or truth are left out of the scores.
  - The `truth_column` is trusted as given: the key does not check that the target was drawn
    from it, so a truth column that is not the target's real probability gives a wrong score.
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
| `cate` | The mean effect per level of each effect modifier. A numeric modifier with more than 10 distinct values is cut into quartiles; one with 10 or fewer gets an entry per value. Each entry gives `level`, `rows` and `effect` |
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

The `adjustment_set` is a minimal set of published columns that blocks every back-door path.
The search uses only published columns that the treatment does not cause, trying the
published parents of the treatment, then the published confounders, their union and every
such column, and keeps the first set that d-separates the treatment from the outcome. A
hidden cause is fine when a published column blocks its paths. The set is `null` when no
published set works, such as a hidden confounder with no published blocker, or a correlation
linking the treatment and the outcome directly (front-door structures have no back-door set).
A cause of the treatment that is correlated with the outcome counts as a confounder.
Adjusting for a mediator or a collider biases the estimate, which is what the example is for.

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

### Class imbalance

`positive_share` sets how rare the positive class of a classification target is:

```json
"question": {
  "target": "churned", "task": "classification",
  "truth_column": "churn_probability", "positive_share": 0.05
}
```

- The `truth_column` must be a float formula column of probabilities that the target reads,
  such as `sigmoid(...)` read by `bernoulli(churn_probability)`.
- The log-odds of every row are shifted by one constant until the probabilities average the
  share. The rows keep their order and every planted dependence; only the rate moves.
  The share is expected, not exact: each label is still drawn.
- It is not available for a regression question, a panel or a treatment, which would
  change the probabilities after the shift. The truth column cannot have `noise_std`,
  `conditions` or `exact`.
- With a `test_shift` the shifted test rows are calibrated to the same share, so a covariate
  drift does not change how rare the positive class is.

The key gains a `class_balance` section: `requested_positive_share`, the `positive_rate` of
the published train and test labels, `majority_class_accuracy` (what always answering the
common class scores), `pr_auc_baseline` (the area under the precision-recall curve of a model
with no skill, equal to the test positive rate) and `best_achievable_pr_auc` (the same curve for
the true probabilities).

### Difficulty levels

`difficulty` on the question (`easy`, `medium` or `hard`) sets four things together:

| Level | Target noise | Extra distractor columns | Label noise (train) | Test drift |
|---|---|---|---|---|
| `easy` | 0.5 × `noise_std` | 0 | none | none |
| `medium` | 1 × `noise_std` | 4 | 2% of labels flipped, or noise of 5% of the target's std | none |
| `hard` | 2 × `noise_std` | 10 | 5% of labels flipped, or noise of 10% of the target's std | the mean of the first published numeric driver rises by 0.5 std |

- Target noise multiplies the `noise_std` of the target's formula. A target without
  `noise_std` is left as it is, so for such a target, typically a classification target drawn
  with `bernoulli`, the level does not lower `best_achievable`; the drift of `hard` can even
  raise it (accuracy 0.759 at `easy` and 0.796 at `hard` on the churn example). Only the scores
  a learned model reaches fall, through the label noise and the extra distractors.
- Distractors are columns named `extra_feature_NN`, cycling between a normal number, a
  four-level category and a count. Nothing reads them, and the key lists them as distractors.
- Label noise is the question's own `label_noise` with the `train` split. Drift is a
  `test_shift` on the first published numeric driver that is drawn with a `mean` and a `std`.
  Drift is skipped for a panel and for related tables.
- A setting the question gives itself, such as its own `label_noise` or `test_shift`, is kept
  and the level does not add another. A setting that does not fit the schema is skipped.
- Without `difficulty` nothing changes. With it, the key gains a `difficulty` section that
  names the level, the noise before and after, the distractor columns, and the label noise
  and drift that were applied or why they were not.

### Task description

`QUESTION.md` is written from the question and from what is published: the title and prompt,
what to predict, the files and row counts, the score, notes on blank cells and a treatment or
panel layout, the rules and the columns. It never reads the key's findings, so it does not
name drivers, formulas, why cells are blank, label noise, drift or aggregates. A rare positive
class is mentioned with the share a participant could measure in the training rows.

`metric` on the question (`rmse`, `mae`, `r2`, `accuracy`, `roc_auc`, `pr_auc` or `log_loss`)
names the score; it must suit the task, and the key then repeats it as `metric`. Without it
the text uses RMSE for regression, ROC AUC for classification and PR AUC when the positive
class is rare (a `positive_share`, or under 20% of the training rows).

## Panels and time series

A `panel` makes the table `entities × periods` rows: the same entities observed over many
periods, such as 40 stores over 36 months. The full example is
[store_sales_panel.json](../examples/store_sales_panel.json).

```json
"panel": {
  "entity": "store_id", "entities": 40,
  "time": "date", "start": "2022-01-01", "periods": 36, "frequency": "MS"
}
```

- The panel creates two columns, `entity` (integer ids 1 to `entities`) and `time` (one date
  per period, spaced by `frequency`, a pandas offset alias such as `MS`, `W` or `D`). They come
  first, and rows are ordered by entity and then by period.
- The row count is fixed by the panel. Leave `rows` out of `synthesize` and `challenge`, or
  pass exactly `entities × periods`; any other number is a `SchemaError`. `--rows` is optional
  on the command line for the same reason.
- A panel is generated from the schema alone, so it cannot be combined with a reference table,
  and it cannot have `constraints`, because dropping rows would break its periods.
- Names the panel uses (`entity`, `time`, `period`, `entity_index`) cannot be column names.

**Levels.** `level` says how often a column's value is drawn:

| `level` | Drawn | Example |
|---|---|---|
| `row` (default) | for each entity and period | weather, sales |
| `entity` | once per entity, repeated over its periods | store size, region |
| `period` | once per period, shared by every entity | a market shock |

An entity-level column can read only entity-level columns, and a period-level column only
period-level columns, so the value stays constant where it should. Row-level columns read
anything. Correlations join columns of the same level. Only entity-level columns can be
`unique`. A period-level column cannot be `exact` or have `outlier_fraction`, and a row-level
column cannot be `exact` in a panel, because every period would hit the target on its own;
the schema is rejected with a `SchemaError`.

**Formulas.** A formula can read `period` (the 0-based period of the row), `entity_index`
(the 0-based entity), the time column (`month(date)`, `dayofyear(date)`) and
`lag(column, k)`:
- `lag(sales)` or `lag(sales, 2)` (also `lag(sales, k=2)`) is the column's value `k` periods
  earlier for the same entity, with `k` a whole number from 1. It is missing for the first `k`
  periods of each entity.
- A formula may lag itself. `0.7 * lag(sales) + 20` is an AR(1) series. Rows are built period
  by period, all entities together, so the value a formula lags is already built.
- Give the first period its own value, since the lag is missing there:
  `where(period == 0, 130, 20 + 0.7 * lag(sales))`.
- `lag` is allowed in formulas only, not in conditions or constraints, and not on entity-level
  columns. A period-level column can lag period-level columns.

**Regimes.** A regime is a condition on `period` (or on the time column):

```json
{"name": "new_competitor", "kind": "boolean", "true_share": 0,
 "conditions": [{"when": "period >= 24", "true_share": 1}]}
```

Formulas that read `new_competitor` change level from period 24 on. The key lists these
conditions under `time_series.regimes`.

### Time-series questions

Add `"split": "time"` to the question to hold out the last `test_share` of the periods
(rounded to at least one) as the test set. The default, `"random"`, picks test rows at random,
so every period appears in both splits. A time split needs a panel and at least one training
period.

- **Drift.** With `split: "time"` and a `test_shift`, the shifted schema generates the test
  periods: a regime change in the future. The same entities carry on, and an autoregressive
  column continues from its last training value. Entity-level columns cannot be shifted. A
  shift on a panel needs a time split.
- **Label noise** applies to the published labels as for any table.
- **Leakage.** A column that reads the current target is a leak, as before. A column that reads
  `lag` of the target is a legitimate feature: the earlier value is known when predicting. The
  key lists such columns under `features.lags_target` instead of `distractors`. In a time
  split, a test row's lag of the target would be the previous test row's withheld label, so
  the test rows after the first test period are published with such columns blank. The first
  test period keeps the lag from the last training period. The task is to forecast the test
  periods from the training horizon, and `time_series.withheld_test_columns` names the blanked
  columns. A random split has no horizon, so a schema whose published columns read an earlier
  target is refused with a random split: hide those columns or use `split: "time"`. This holds
  when the lag reads a column that depends on the target, even a hidden near copy of it.
- **Causal questions.** A `treatment` on a panel is refused with a `SchemaError`. A treatment
  reaches later periods through lags, and the effect replay recomputes descendants within one
  row, not across periods, so its effects would be wrong.
- **Best achievable.** The truth of a target formula that lags is its formula evaluated
  again without noise on the observed earlier values, the best one-step-ahead prediction. With
  the lag columns withheld after the first test period, a forecast from the training horizon
  cannot reach it.

The key gains a `time_series` section:
- `entity`, `time`, `entities`, `periods`, `start`, `frequency` and `split`;
- `periods_per_split`: the number of periods in `train` and in `test`;
- `first_test_period`: its index and date, for a time split (null for a random split);
- `lags`: for each column, the columns its formula lags and by how many periods;
- `withheld_test_columns`: columns that read an earlier target and are blank after the first
  test period of a time split;
- `regimes`: each condition on `period` or the time column, with the column it changes.

## Related tables

A schema with a `tables` list describes several tables joined by foreign keys. The full
example is [examples/customers_orders_challenge.json](../examples/customers_orders_challenge.json).

```json
{
  "tables": [
    {"name": "customers", "key": "customer_id", "columns": [...], "aggregates": [...]},
    {
      "name": "orders", "key": "order_id",
      "parent": "customers", "foreign_key": "customer_id",
      "rows_per_parent": {"distribution": "poisson", "mean": 5},
      "columns": [...]
    }
  ]
}
```

- The first table is the root and has as many rows as `--rows` asks for. Every other table
  names its `parent` (a table listed before it), the `foreign_key` column that holds the
  parent's key, and `rows_per_parent`: a distribution (Poisson by default, or any numeric
  distribution with `mean`, `std`, `minimum` and `maximum`) for how many rows each parent
  gets. A parent can have none.
- Keys are generated and are the whole numbers from 1, so every foreign key points at a row
  that exists. Neither key is listed in `columns`.
- A child column can read its parent's columns as `<parent table>_<column>`, such as
  `customers_loyalty_tier`, in formulas and conditions. A grandchild reads its own parent.
- Tables cannot have a panel or constraints. Missing-value mechanisms, correlations, formulas
  and conditions work inside each table as usual.

**Aggregates.** A table can summarise a child table's rows per row of its own with
`aggregates`: `{"name": "total_spend", "table": "orders", "function": "sum", "column":
"amount", "when": "not refunded"}`.
- `function` is `count` (no `column`), `sum`, `mean`, `min` or `max`. `when` keeps only the
  child rows where an expression over the child's columns is true. A parent with no matching
  rows gets 0.
- Other columns of the parent can read an aggregate by name, such as
  `sigmoid(-0.4 * order_count)`. Those columns, and every column that reads them, are built
  after the child rows, so a child can only read parent columns that do not depend on an
  aggregate. A table needs at least one column that does not.
- Aggregates are withheld from the published tables.

**Challenges.** Put the `question` at the top level of the schema, with the target in the
root table. `oblitus challenge` splits the root table into `train.csv` and `test.csv` and
writes every other table in full, as `<table>.csv`, so a participant can build the aggregates
that the withheld ones stand for. A target that reads an aggregate cannot be predicted from
the root table alone.

The key adds:
- `tables`: for each table its rows, key and, for a child, its parent, foreign key and
  `rows_per_parent` (declared, and the observed mean and maximum), plus `missingness`;
- `child_aggregates`: the withheld aggregates, those `the_target_depends_on` (through other
  columns too) and those `not_used_by_the_target`.

`oblitus synthesize --schema ... --rows N -o FOLDER` writes one CSV per table into the folder.
In Python, `Anonymizer.synthesize_tables(schema, rows)` returns the tables and the aggregates.

**Refusals.** A causal question (`treatment`) and a `test_shift` are refused with a
`SchemaError`: both need the build replayed from stored random numbers, which each table keeps
for itself but not across tables, and a treatment on a parent would also change its children.
Difficulty skips drift for the same reason.
