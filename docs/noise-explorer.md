# Noise explorer

```bash
uv run anonymize explore-noise customers.csv --column salary -o salary_noise.html
```

```python
from pathlib import Path

from data_anonymizer.noise_explorer import explore_noise, save_page

exploration = explore_noise(customers, "salary", seed=1)
exploration.metrics  # one row per technique and strength
save_page(exploration, Path("salary_noise.html"))
```

The page is one HTML file with no external scripts. It runs every noise technique on the
column at several strengths:

| Technique | Strengths, weakest to strongest |
|---|---|
| `gaussian_noise` | relative scale 0.02 to 1.5 |
| `multiplicative_noise` | relative scale 0.02 to 0.5 |
| `laplace_noise` | ε 100 to 0.5, with the column's own bounds |
| `rank_swap` | maximum distance 0.5% to 40% of the rows |
| `microaggregate` | group size 2 to 50 |

Pick a technique and slide its strength. Clicking a point on the risk map selects that
setting.

## What it shows

**Tiles**
- Mean change and spread change.
- Typical record shift: the median |released − true|, in standard deviations.
- Re-linked records, with the chance rate for comparison.
- Values still exposed.
- Kolmogorov–Smirnov distance.

**Charts**
- **The distribution moves this much**: original against released histogram, on one
  axis per technique.
- **Each record moves this much**: true against released value for up to 1,500 sampled
  records. The shaded band is ±`--band` standard deviations (default 0.1). Points inside
  it are records whose true value an attacker can still pin down.
- **Risk against information loss**: for every technique and strength, re-linked
  records (log scale) against typical record shift (log scale). Lower and further left is
  better, and the dashed line is chance.

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
