# Noise explorer

The noise explorer shows what each noise technique would do to one numeric column before you
put it in a plan. You write a page for the column, open it in a browser, and compare techniques
and strengths by how far records move against how many can still be re-linked.

## Write the page

From the command line, with a table that has a numeric column:

```bash
uv run anonymize explore-noise customers.csv --column monthly_spend -o monthly_spend_noise.html
```

It prints `wrote monthly_spend_noise.html; open it in a browser`. To try it without your own
data, first write a sample table from the example schema:

```bash
uv run anonymize synthesize --schema examples/customers_schema.json --rows 2000 -o customers.csv
```

From Python:

```python
from pathlib import Path

from data_anonymizer.noise_explorer import explore_noise, save_page

exploration = explore_noise(customers, "monthly_spend", seed=1)
exploration.metrics  # one row per technique and strength
save_page(exploration, Path("monthly_spend_noise.html"))
```

## Open the page

The page is a single HTML file. It needs no server and no network, so open it like any local
file, in a current Chrome, Edge, Firefox or Safari:

| System | Command |
|---|---|
| Linux | `xdg-open monthly_spend_noise.html` |
| macOS | `open monthly_spend_noise.html` |
| Windows (PowerShell or cmd) | `start monthly_spend_noise.html` |
| WSL | `explorer.exe "$(wslpath -w monthly_spend_noise.html)"` |

Double-clicking the file in a file manager works too, as does dragging it onto a browser
window. The page follows the system's light or dark theme.

**The page holds real values.** It embeds histograms of the column and up to 1,500 of its
original values, for the per-record chart. Treat the file like the original table: keep it
where the data may be, and do not attach it to a ticket, chat or public repository.

## What the page runs

It runs every noise technique on the column at several strengths:

| Technique | Plan parameter | Strengths, weakest to strongest |
|---|---|---|
| `gaussian_noise` | `relative_scale` | 0.02 to 1.5 |
| `multiplicative_noise` | `relative_scale` | 0.02 to 0.5 |
| `laplace_noise` | `epsilon` | 100 to 0.5, with the column's own bounds |
| `rank_swap` | `max_distance_percent` | 0.5 to 40 (% of the rows) |
| `microaggregate` | `group_size` | 2 to 50 |

A chosen setting goes into the plan as, for example:

```json
{"column": "monthly_spend", "technique": "rank_swap", "params": {"max_distance_percent": 5.0}}
```

Columns with more than 200,000 numeric values are explored on a random sample of that
size (`--max-rows` or `max_rows=` to change it); the page says so. The metrics are shares
and medians, which a sample this size estimates closely.

Pick a technique and slide its strength. Clicking a point on the risk map selects that
setting. A good setting sits low and to the left on the risk map: few re-linked records for
the shift you accept. Copy its technique and strength into the plan, as above.

## What it shows

**Tiles**
- Mean change and spread change.
- Typical record shift: the median |released − true|, in standard deviations.
- Re-linked records, with the chance rate for comparison.
- Values still exposed.
- Kolmogorov–Smirnov distance.

**Charts**
- **The distribution moves this much**: original against released histogram, on one
  axis per technique. The axis covers the middle 99% of the original and released values,
  so one extreme outlier does not squash the rest. Values beyond it are counted in the end
  bands, whose labels read "or less" and "or more", and a note gives their share.
- **Each record moves this much**: true against released value for up to 1,500 sampled
  records. The shaded band is ±`--band` standard deviations (default 0.1). Points inside
  it are records whose true value an attacker can still pin down.
- **Risk against information loss**: for every technique and strength, re-linked
  records (log scale) against typical record shift (log scale). Lower and further left is
  better, and the dashed line is chance. A log scale has no zero, so settings with no
  re-linked records or no shift sit on the edge of the axis, at the tick labelled 0.

**Table**: every strength of the selected technique, with all metrics.

## The metrics

| Metric | Meaning |
|---|---|
| Re-linked records | Share of released values whose nearest original value belongs to the same record (distance-based record linkage). Ties split the credit, so microaggregation with groups of k scores 1/k. Chance is 1 / rows |
| Values still exposed | Share of records whose true value lies within ±band standard deviations of the released value (interval disclosure) |
| Typical / 95th percentile shift | How far records moved, in standard deviations |
| Wasserstein / IQR | Distance between the distributions, in units of the original interquartile range |
| KS | Kolmogorov–Smirnov statistic: 0 means identical distributions, 1 disjoint |

The same linkage measure decides the anonymization check for noisy columns: re-linking
more than 1 record in `k_target` fails it.
