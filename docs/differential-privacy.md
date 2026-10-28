# Differential privacy

Differential privacy (DP) is a promise about a release, not about a table: whether or not any one
person's record is in the data, the release looks almost the same. The `oblitus.differential_privacy`
package offers two releases that keep this promise, both under a total privacy budget that it
spends, itemizes and refuses to exceed:

- noisy **counts, sums and means per group** (`Anonymizer.aggregate_private`, `oblitus dp-aggregate`),
  computed with [OpenDP](https://opendp.org);
- a **synthetic table** (`Anonymizer.synthesize_private`, `oblitus dp-synthesize`), built with the MST
  algorithm, or with AIM (`oblitus[aim]`) when relations between three or more columns matter.

## What the privacy budget means

The budget is written epsilon (and delta). A small epsilon means the release changes little
whether or not one person is in the data, so it reveals little about that person. A large epsilon
means more accuracy and weaker protection.

| epsilon | In practice |
|---|---|
| 0.1 to 1 | Strong protection. Expect clearly noisy small groups and weak pair structure. |
| 1 to 10 | A common range for published statistics. Large groups stay accurate. |
| 50 and above | Little protection. The noise is small enough to be tested against, not trusted. |

Delta is the chance that the epsilon promise fails. Choose it far below one over the number of
records, such as `1e-6`. A budget with only epsilon (no delta) is a pure budget: it gives the
strictest promise, and it only accepts Laplace noise.

DP protects against **any** attacker, including one who knows everything about every other
record and has unlimited computing power. The price is accuracy: the noise is real, and it cannot be
tuned away without spending more budget.

The library keeps its accounts in zero-concentrated DP (zCDP), where the cost of each noisy
step is a number `rho` and costs simply add. `PrivacyBudget.from_epsilon_delta(epsilon, delta)` converts
to the matching rho with `epsilon = rho + 2 * sqrt(rho * ln(1 / delta))`, and the report converts the rho
spent back to an epsilon at the same delta, so it never claims more than was declared.
`PrivacyBudget.from_rho(rho)` declares the budget in rho directly.

```python
from oblitus import PrivacyBudget

pure = PrivacyBudget.pure(1.0)  # epsilon = 1, Laplace noise only
approximate = PrivacyBudget.from_epsilon_delta(1.0, 1e-6)
```

## Bounds and domains must be public

Every release needs two kinds of facts that you declare, and the library never reads them from the data:

- **The groups, or the values of a category column.** A list of group values you read from the data
  would reveal which values exist. List them from public knowledge (every region, every plan, every month).
  Every listed group is released, including empty ones.
- **Lower and upper bounds of a number.** A value outside the bounds is clipped to them, which caps
  how much one person can move a sum. Taking the minimum and maximum from the data would leak
  the extreme records. Choose bounds from the meaning of the column (an age is 0 to 120).

A value in a group column that is not in the declared list is an error, never silently dropped. Missing
values are an error too: fill them with a declared value first. Bounds that are too wide add noise,
and bounds that are too tight clip real values and bias the result.

## Aggregates

```python
from oblitus import Anonymizer, PrivacyBudget, Statistic

result = Anonymizer().aggregate_private(
    orders,
    {"city": ["north", "south", "east"]},
    [
        Statistic(name="customers", kind="count"),
        Statistic(name="spend_total", kind="sum", column="spend", lower=0, upper=500),
        Statistic(name="spend_mean", kind="mean", column="spend", lower=0, upper=500),
    ],
    PrivacyBudget.from_epsilon_delta(1.0, 1e-6),
)
result.table  # one row per group: the noisy statistics
result.accuracy  # the 95% interval half-width of every value in `result.table`
result.budget  # what was spent, per statistic
```

Install the extra first: `pip install "oblitus[opendp]"`. The package turns on OpenDP's contributed
features itself, and imports it only when called.

- The budget is split over the noisy queries: a count and a sum take one share each, and a mean
  takes two (a sum and a count). A record is in exactly one group, so every group is released with the
  same noise and costs each statistic's share once, however many groups there are.
- A pure budget uses Laplace noise, a budget with delta or rho uses Gaussian noise.
- `accuracy` is the half-width within which the noisy value is the exact one plus or minus
  that distance, with probability `confidence` (default 0.95). The interval of a mean divides the
  noisy sum and count and is a first-order approximation, which is loose when the noisy count is near zero.
- Noisy counts can be negative and noisy means are kept inside the declared bounds. Clipping a count
  at zero would bias sums of counts, so it is left to you.

The command line takes the same declaration in a JSON file:

```json
{
  "group_by": {"city": ["north", "south", "east"]},
  "statistics": [
    {"name": "customers", "kind": "count"},
    {"name": "spend_mean", "kind": "mean", "column": "spend", "lower": 0, "upper": 500}
  ]
}
```

```bash
oblitus dp-aggregate orders.csv --spec spec.json --epsilon 1 --delta 1e-6 -o noisy.csv \
  --accuracy-output accuracy.csv
```

## Synthetic data with MST

```python
result = Anonymizer().synthesize_private(
    customers,
    {
        "age": {"kind": "numeric", "lower": 0, "upper": 100, "bins": 10, "integer": True},
        "plan": ["free", "pro", "team"],
    },
    PrivacyBudget.from_epsilon_delta(2.0, 1e-6),
    rows=5_000,
)
result.data  # the synthetic table
result.budget  # the itemized spends and the epsilon reached
result.tree_edges  # the column pairs whose joint distribution was measured
```

MST (McKenna, Sheldon and Miklau, 2021) learns only low-dimensional summaries, so the noise stays
small enough to be useful:

1. Spend a third of the budget on noisy one-way counts of every column (Gaussian noise).
2. Spend a third on choosing a spanning tree of the columns. Each of the `columns - 1` edges is
   picked by the exponential mechanism, preferring pairs whose joint counts differ most from
   independence. The score is the distance between a pair's counts and the independence
   model built from the noisy one-way counts, which changes by at most one when one record changes.
3. Spend a third on noisy two-way counts of the chosen pairs (Gaussian noise).
4. Combine the noisy numbers into one estimate per column, fit non-negative pair tables whose margins
   match those estimates, and sample rows from the resulting tree of conditional distributions.
   This step only reworks numbers that are already noisy, so it costs no privacy.

Each numeric column is cut into the bins you declare, and a synthetic value is drawn uniformly
inside its bin, so the table is as precise as your bins. Without `rows`, the table has the noisy row count; passing `rows` is free because the row
count you ask for is a choice of yours, not a statistic of the data.

What to expect:

- Only the chosen pairs keep their dependence. A relation between two columns that the tree
  does not connect is only as strong as the path between them implies, and three-way effects are lost.
- A tight budget leaves mostly the one-way distributions. Raise epsilon, use fewer bins, or drop
  columns, before concluding that the method fails.
- Large joint domains are refused: a pair of columns with more than 1,000,000 cells is an error.

The command line reads the declared domains from JSON (`column: list of values`, or
`{"kind": "numeric", "lower": 0, "upper": 100, "bins": 10, "integer": true}`):

```bash
oblitus dp-synthesize customers.csv --domains domains.json --epsilon 2 --delta 1e-6 \
  --rows 5000 -o private.csv
```

It prints the epsilon spent and the pairs measured.

## Synthetic data with AIM

AIM (McKenna, Mullins, Sheldon and Miklau, 2022) is the iterative successor of MST. MST measures
one tree of column pairs, chosen once. AIM measures in rounds and puts the budget where the model
is worst. It also measures groups of three or more columns, when the *workload* (the column
groups the synthetic data should reproduce) lists them. It needs the `aim` extra, which installs
Private-PGM (`mbi`) and JAX, about 350 MB, without PyTorch:

```bash
pip install "oblitus[aim]"
```

```python
result = Anonymizer().synthesize_private(
    patients,
    domains,
    PrivacyBudget.from_epsilon_delta(2.0, 1e-6),
    method="aim",
    workload=[("age", "region", "diagnosis")],  # every pair when omitted
)
result.measured_marginals  # every column group whose counts were measured
```

1. Spend a small share of the budget on noisy one-way counts of every column.
2. Each round, pick with the exponential mechanism the workload group (or a sub-group of one)
   whose counts the current model gets most wrong. The score is the weighted L1 error, less the
   error the measurement noise alone would cause. Measure that group's counts with Gaussian
   noise, and refit the model to every measurement so far with Private-PGM.
3. When a measurement barely changes the model, halve the noise of later rounds. The last round
   spends whatever is left, so the report always shows the whole declared budget spent.
4. Sample rows from the fitted model. This costs no privacy.

The mechanism (choices, noise and accounting) lives in oblitus and draws from your generator.
Private-PGM only fits a model to numbers that are already noisy. Domains are your declared ones,
and every round's choice and measurement is a line of the budget report.

What to expect:

- AIM is slower than MST: it refits the model every round, about 16 rounds per column at most.
  A table of 10 columns takes minutes, not seconds.
- With the default workload of pairs, AIM usually matches or beats MST at the same budget.
  It is clearly better when a relation only shows in three columns at once (a value that depends
  on the combination of two others). MST loses such a relation.
- Workload groups with more than 10,000 cells are left out, because their counts would drown in
  the noise. The model stops growing at 10,000,000 cells (`max_model_cells` in
  `synthesize_aim`), which bounds memory.

On the command line, add `--method aim`, and `--workload` once per group:

```bash
oblitus dp-synthesize patients.csv --domains domains.json --epsilon 2 --delta 1e-6 \
  --method aim --workload age,region,diagnosis -o private.csv
```

## Budget accounting

`compose_budget(budget, spends)` is the single place where costs are added. Every noisy step is a `Spend`
(a Laplace or Gaussian mechanism, or one exponential-mechanism choice), and the composition:

- adds the rho of every step (an epsilon-DP step costs `epsilon² / 2`, an exponential-mechanism choice
  `epsilon² / 8`, a Gaussian step with sensitivity `s` and scale `sigma` costs `s² / (2 sigma²)`);
- raises `PrivacyBudgetError` if the total is above the declared budget, and, under a pure budget,
  if a step is not pure epsilon-DP;
- returns a `BudgetReport` with the declared epsilon, the rho and epsilon spent, and every step.

The releases above build their spends and compose them before returning. Each separate call
spends its own budget: two releases of the same data cost the sum of their budgets, and keeping that total
within what you promise is up to you.

## Limits

- The privacy unit is one record added or removed. If one person has many records, either
  aggregate to one record per person first, or divide epsilon by the most records one person can have.
- MST and AIM draw their noise with numpy's floating-point generators. Floating-point noise can in
  principle leak through the least significant bits of the result (Mironov, 2012). The OpenDP aggregates
  use OpenDP's hardened noise. Prefer the OpenDP path for high-stakes releases.
- A refusal is not private. A missing value, a value outside a declared domain or text in a
  numeric column stops the release with an error that names the column and the kind of
  problem, never the value. Whether the call succeeds still tells whoever sees it that some
  record breaks the schema, so clean the data and keep errors inside the data owner's process.
- A fixed `seed` makes the noise reproducible, which lets anyone who knows it subtract the
  noise. Use a seed for tests and demos only. Without a seed, the noise is fresh.
- AIM turns on JAX's double precision (`jax_enable_x64`) and turns off its compilation cache
  when it first loads `mbi`. Both are process-wide JAX settings, so other JAX code in the same
  process sees them.
- oblitus implements AIM's mechanism itself rather than through `smartnoise-synth`. That
  package pulls in PyTorch, learns each column's domain from the data, and keeps the budget
  accounting where oblitus cannot itemize or refuse it.
