# Differential privacy

Differential privacy (DP) is a promise about a release, not about a table: whether or not any one
person's record is in the data, the release looks almost the same. The `oblitus.differential_privacy`
package offers two releases that keep this promise, both under a total privacy budget that it
spends, itemizes and refuses to exceed:

- noisy **counts, sums and means per group** (`Anonymizer.aggregate_private`, `oblitus dp-aggregate`),
  computed with [OpenDP](https://opendp.org);
- a **synthetic table** (`Anonymizer.synthesize_private`, `oblitus dp-synthesize`), built with the MST
  algorithm.

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
- MST draws its noise with numpy's floating-point generators. Floating-point noise can in
  principle leak through the least significant bits of the result (Mironov, 2012). The OpenDP aggregates
  use OpenDP's hardened noise. Prefer the OpenDP path for high-stakes releases.
- A fixed `seed` makes the noise reproducible, which lets anyone who knows it subtract the
  noise. Use a seed for tests and demos only. Without a seed, the noise is fresh.
- AIM, the iterative successor of MST, is not included. Its reference implementation
  (`smartnoise-synth` on top of `mbi`) installs on Python 3.12, but pulls in PyTorch and JAX
  (several gigabytes), learns each column's domain from the data rather than from a public declaration,
  and spends its budget inside the library, where oblitus cannot itemize or refuse it. It stays on
  the roadmap until it can run on public domains with the budget reported here.
