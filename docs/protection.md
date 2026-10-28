# Pseudonymization and anonymization

Every plan has a **protection** target and a **fidelity**.

| | `statistical` fidelity | `privacy` fidelity |
|---|---|---|
| **`pseudonymization`** | Realistic fakes, consistent per person. Light perturbation. Joins and distributions survive | Keyed hashes for identifiers, generalized quasi-identifiers, k-anonymity enforced |
| **`anonymization`** | Random fakes per cell. Quasi-identifiers generalized only as far as k-anonymity needs. Means kept | Identifiers suppressed. Quasi-identifiers generalized. Sensitive values randomized |

**Pseudonymization** replaces identifiers with stand-ins that the key holder can re-link
or reveal. Under GDPR Article 4(5) and the LGPD, the output is still personal data. Its
value is that one dataset can be joined to another, and an authorized team can get back
to the person.

**Anonymization** removes every link to people that the plan's techniques can remove:
- A plan that names a linking technique is refused before anything runs. That covers
  keyed hashes, Argon2 and bcrypt, consistent surrogates, scrambling and encryption.
  Argon2 and bcrypt with a random salt are refused too, because anyone can test a
  guessed value against the published hash.
- The plan runs under a throwaway key for everything that could link a release: category
  codes and random fakes are fresh in every run, so two releases cannot be joined on them.
  Noise stays keyed on the secret key and seed. Reissuing the same release gives the same
  noise, so averaging repeated releases gains nothing.
- Quasi-identifiers are generalized, and local suppression then blanks cells until each
  combination is shared by at least `k_target` rows. Rows whose quasi-identifiers are all
  blank still form a group; when they number fewer than `k_target`, the smallest complete
  group is blanked too, so no row has to be dropped.
  Mondrian generalization is the other way to reach `k_target`; see
  [Choosing how k-anonymity is enforced](#choosing-how-k-anonymity-is-enforced).

The result is then checked.

## Default proposals

Every direct identifier follows the row for its protection and fidelity:

| | `statistical` | `privacy` |
|---|---|---|
| pseudonymization | `surrogate` (realistic, consistent); `keyed_hash` for record IDs and identifiers without a fake generator | `keyed_hash` |
| anonymization | `surrogate` with `consistent: false` (random per cell); `scramble_categories` for record IDs, so the rows of one entity stay together without the real ID | `suppress` |

The default target is `anonymization`. Free text with personal data gets `suppress` under
anonymization, because no list or pattern finds every name written into a sentence (a
lowercase or foreign name passes them all), and `redact_entities` under pseudonymization.
Keeping a redacted copy under anonymization is a choice you make in the plan; the release then
stays `at_risk`. Company names get `scramble_categories` (`ORG-…` codes) in every mode.

`scramble_categories` codes are random per run under anonymization. Under
pseudonymization they are keyed (`keyed: true`), so the same label gets the same code in
every release.

Under anonymization, the proposal also gives every untreated text column with at least 20
distinct values that group rows a `scramble_categories` action with prefix `KEY`. The codes are
fresh for every release and keep the grouping, so a model can still use the column while nobody
can look up their own value. Whole-number columns are kept, because they may be measurements
such as counts or hours whose distribution must survive; when one of them groups rows like a
key, the protection report flags it and the release is `at_risk`. Replace it or list it in the
plan.

Two more steps read the whole table under anonymization:
- A timestamp or number left untreated with a different value in nearly every row (90% or
  more distinct) is generalized, since it would single out rows. Statistical fidelity cuts
  timestamps to the month and rounds numbers to three significant figures, which keeps their
  distribution. Privacy fidelity cuts to the year and uses quintile bands, and adds both
  columns to the quasi-identifiers.
- Every other column with at most 50 distinct values joins the quasi-identifiers, so local
  suppression covers the same columns the protection report uses when it looks for unique
  rows.

Pseudonymization proposals are unchanged. CSV files are read with ISO dates parsed as dates.

The proposal picks an entity column: the company column if there is one, otherwise the
record ID. `date_shift` then moves every date of one entity by the same offset, so the
intervals between that entity's events survive.

The other categories:

| Category | pseudonymization, statistical | pseudonymization, privacy | anonymization, statistical | anonymization, privacy |
|---|---|---|---|---|
| url | `surrogate` | `keyed_hash` | random `surrogate` | `suppress` |
| postal_code | keep 3 characters | keep 2 | keep 3 | keep 2 |
| city | `keep` | `suppress_rare` below 10 | `keep` | `suppress_rare` below 10 |
| date_of_birth | `date_shift` ±30 days, per entity | year only | year only | year only |
| other dates (a visit, a purchase) | `keep`, counted as quasi-identifier | `keep`, counted | month only | year only |
| age | `microaggregate` groups of 5 | decades | `microaggregate` groups of 5 | decades |
| gender | `keep` | `keep` | `keep` | `keep` |
| geo_coordinate | Gaussian noise about 1 km | 1 decimal | 2 decimals | 1 decimal |
| income | `rank_swap` within 5% | quintiles | `microaggregate` groups of 5 | quintiles |
| health, special category | `keep` | `randomized_response` p = 0.75 | `keep` | `randomized_response` p = 0.75 |
| nationality | `suppress_rare` below 5 | below 20 | below 5 | below 20 |
| marital status | `keep` | `keep` | `keep` | `keep` |
| business segment | `scramble_categories`, segments under 3 rows merged | under 5 | under 3 | under 5 |
| floor area | quartile bands | quartile bands | quartile bands | quartile bands |
| business revenue, rent | `normalize` robust | `normalize` robust | `normalize` index by entity (rank without one) | `normalize` index by entity (rank without one) |

Any other sensitive categorical column is kept with statistical fidelity and gets
`randomized_response` with privacy fidelity.

k-anonymity is enforced in every mode except pseudonymization with statistical fidelity.
When a technique cannot handle a column's storage kind, such as dates stored as text,
the proposal falls back:
- anonymization: `suppress`;
- pseudonymization: `keyed_hash` for identifiers, and `keep` with "choose a technique by
  hand" for everything else.

## Tokenization vaults

`tokenize` is pseudonymization: the vault maps each token to its value, so anyone who can
read the vault can re-link every record. Treat the vault as you treat the original data:
keep it away from the recipient of the released table, restrict who may read it, and delete it
when no one needs to reveal tokens any more, which cuts the link. The tokens themselves
carry no information about their values. See
[techniques.md](techniques.md#tokenization-and-its-vault). `oblitus apply`, `reveal` and
`apply-many` take `--vault PATH`, a SQLite file.

## The protection checks

`result.protection` holds the outcome:
- `level`;
- `reached`, which is true when the level meets the plan's target;
- `problems`;
- `caveats`.

The checks follow the three criteria of the EDPB Guidelines 02/2026 on anonymisation.

| Criterion | Check |
|---|---|
| No record isolation | Every quasi-identifier combination is shared by at least `k_target` rows (default 5). No row is unique across its discrete released columns, meaning those with at most 50 distinct values, excluding sensitive attributes. No column with at least 20 rows and 90% distinct values is released unchanged, since such a column is a fingerprint. A text column that a technique does not replace (such as `partial_mask`, or `redact_entities` on JSON) counts the same way. An untreated text, integer or whole-number float column (at least 99% of present values whole) outside the plan's lists, with at least 20 distinct values that group rows, is reported too, however many rows each value has: statistics cannot tell an account or card key from a product code, and anyone who knows their own value finds their rows. Pairs of such columns (the 10 with the most distinct values) are checked too: a pair with at least 100 distinct joint values and at most 10 rows on the typical pair is reported as a key that works only together. Replace the column (`scramble_categories` gives fresh codes that keep the grouping) or list it in the plan |
| No linkage | No action uses a linking technique. For noise and perturbation columns, nearest-value matching re-links at most 1 record in `k_target`: no better odds than k-anonymity already accepts. The same limit applies to all perturbed numeric columns taken together, matching each released vector to its nearest original vector on standardized columns. `randomized_response` on a plan with an `entity_column` needs `by`, so a person's rows share one draw |
| No inference | Within each quasi-identifier group, every sensitive attribute takes at least `l_target` distinct values (default 2). When the plan sets `t_target`, no group's distribution of a sensitive attribute may be farther than that from the whole table's (earth mover's distance). Under anonymization, no group may let an attacker guess a sensitive value with high confidence: one value with a share of at least `inference_confidence` (default 0.9) in the group and at least `inference_gain` (default 0.3) above its share in the whole table (see [Skewed groups](#skewed-groups)). A sensitive column released entirely blank is skipped. A numeric sensitive column still in its original units must not let a segment's dominant contributor be read off (dominance and p% rules, see [business-confidentiality.md](business-confidentiality.md)) |

| Level | Meaning |
|---|---|
| `anonymized` | No linking technique and no direct identifier left, and every check above passed on this release: the quasi-identifier groups (including the columns of a joint microaggregation) meet `k_target`, no row is unique across its discrete columns, no column is a fingerprint, noisy values do not re-link by nearest value, and no sensitive attribute fails the diversity, closeness or skew limits. For `apply_many`, the tables are joined through every link and pass the same k and inference checks. It means no check found a way to single out a record or read a sensitive value from this release. It does not mean nobody can be re-identified, and it makes three promises smaller than they may sound: **membership** is not hidden (k-anonymity does not say whether a person is in the table, and Mondrian cut values and medians come from the data, so in a small table they can show whether a particular person is in it; use differential privacy, `synthesize_private` or `aggregate_private`, when membership itself is secret); **inference** is bounded, not excluded (the skew check flags a group at 90% confidence and 30 points above the base rate by default, so a group at 89% passes; lower `inference_confidence` and `inference_gain`, or set `t_target`, for a stricter release); and **people** are counted as entities only when the plan has an `entity_column`, otherwise repeated rows of one person count as several. See the caveats below |
| `at_risk` | No link to people, but a record can still be isolated or a sensitive value inferred. A column cleaned with `redact_entities` always stays here, with or without the `ner` extra: no list, pattern or model finds every name written into a sentence or a JSON field, so the problem suggests dropping the column or reviewing its text. Under anonymization the proposal drops free text for that reason |
| `pseudonymized` | Identifiers replaced, but a linking technique lets the key holder re-link them |
| `identifiable` | A direct identifier was left as it was. This is checked on the data: more than 1% of a direct identifier's values surviving unchanged counts, whatever the technique. For `redact_entities`, any text that still holds a recognisable document, email, phone, listed first name with a surname, or municipality counts |

A column rescaled only by shift and scale (`normalize` with `robust`, `zscore` or `minmax`
and no `by`, whatever `allow_shape_change` says) is a problem under anonymization, not a
caveat: two known values, or one known value plus a true total, undo the scale. Group by the
entity with `allow_shape_change`, or use `rank`.

The `anonymized` level also comes with caveats, because these checks are necessary but
not sufficient. They only look at the columns the plan lists and at tables released in one
call:
- columns the detector missed are not treated;
- free text can hide names, since no list, pattern or model finds every name, which is why
  anonymization drops free text by default and keeps a redacted copy at `at_risk`;
- quasi-identifiers outside the plan, or outside data an attacker holds, are not counted;
- whether the output is anonymous for a given recipient depends on the context. The CJEU
  ruled in *EDPS v SRB* (C-413/23 P, 2025) that pseudonymized data may be anonymous for a
  recipient who has no means to re-identify it.

A contextual assessment, such as the ICO's motivated-intruder test or the ANPD's risk
assessment, is still required.

## Choosing how k-anonymity is enforced

With `enforce_k_anonymity: true`, the plan's `k_anonymity_method` picks the way:

| `k_anonymity_method` | What it does | Costs |
|---|---|---|
| `local_suppression` (the default) | Blanks quasi-identifier cells in rows of small groups, and nothing else | Blank cells, but the other rows keep their exact values |
| `mondrian` | Cuts the rows into groups of at least `k_target` and replaces every quasi-identifier cell with its group's description | No blank cells, but every quasi-identifier column becomes coarser |

Both run on the released table after the column actions, so they also cover columns the
actions left alone. Either way, the table is then measured and the protection checks run on it.

### Mondrian generalization

Mondrian follows LeFevre, DeWitt and Ramakrishnan (2006), in its strict form:
1. All rows start as one group.
2. The group is cut at the median of the quasi-identifier that spreads the most across it,
   relative to the whole table. Rows with the same value on that column stay on one side, so a
   cut never separates equal values.
3. A cut is allowed only when both halves keep at least `k_target` rows. If the widest column
   cannot be cut, the next widest is tried. A group that no column can cut is final.
4. Every cell of a final group gets the same description of the group.

How a group is described, per column type:

| Column | `range_style: "range"` (default) | `range_style: "midpoint"` |
|---|---|---|
| Number | `20 to under 40`; the outermost groups are open-ended: `under 20`, `90 or more` | The group's median row value, as a float |
| Date or timestamp | `2000-01-01 to before 2001-01-01`, `before ...`, `... or later`, in ISO form | The group's median date, in the column's own type |
| Category | The set `{A, B, C}`, or the value when all are equal. With a hierarchy for the column in the plan's `hierarchies`, the closest common ancestor instead | Same as `range` |

The description comes from the cuts that enclose the group, not from the smallest and largest
value inside it, so one extreme record (a single 90-year-old among teenagers) never shows in
the release. A column that no cut bounds reads `any value`, unless every value in the table is
equal. A column with one value in a group is still described by its bounds. Categories are
described by the set of values present. A category held by fewer than `k_target` rows in the
whole table is never named: it reads `other` inside the set, so the set does not show that its
few holders are in the table.

The cut values and the medians are values of the data, so in a small table they can show whether
a particular person is in it (a cut at `under 50` where the nearest ordinary values are 20 and
60). Mondrian protects against singling a person out of the released table, not against telling
whether that person was in the source. Use differential privacy when membership is secret.

A missing cell takes its group's description, so that every row of a group holds the same
value. A group in which the column is missing everywhere keeps it missing. A column of dates
stored as text is a category.

A hierarchy also decides which categories sit next to each other when the group is cut, so a
cut separates Southeast from Northeast states and not at random:

```json
{
  "quasi_identifiers": ["age", "cep", "gender"],
  "enforce_k_anonymity": true,
  "k_anonymity_method": "mondrian",
  "range_style": "range",
  "hierarchies": {"cep": {"preset": "cep"}}
}
```

Hierarchy forms and the Brazilian postal code preset are described in
[techniques.md](techniques.md#generalization-hierarchies).

The cut works on the distinct quasi-identifier combinations with their row counts, so it
gives the same answer on pandas, Polars and DuckDB, and its cost depends on how many
distinct combinations there are, not how many rows. A table with fewer than `k_target` rows
becomes one group, which cannot meet the target; the protection check says so. Because the
output of a numeric or date column is text under `range`, a model that needs the column as a
number should use `midpoint`.

### Joint microaggregation

The plan's `joint_microaggregation` lists sets of numeric columns to replace together:

```json
{"joint_microaggregation": [{"columns": ["height", "weight"], "group_size": 5}]}
```

Records are grouped by their distance over all the listed columns, standardized so none
dominates, with the MDAV method (Domingo-Ferrer and Torra, 2005). Every group holds at least
`group_size` records, and each column holds its group's mean, so every released vector is
shared by at least `group_size` records and every column keeps its mean. The step runs after
the column actions and before k-anonymity is enforced, on the whole table, and it does not
change a column's dtype: integers are rounded back to integers.

A record with a missing value has no position in the full space, and keeping its other values
would single it out. Records are grouped by which listed columns they have, and each such set
is microaggregated over the columns it shares, in groups of at least `group_size`. A set with
fewer records than that has its listed columns blanked. When the blanked records are still
fewer than `group_size`, the smallest groups are blanked with them, so no blank record stands
alone. The protection check treats the listed columns as quasi-identifiers: it counts them when it
looks for records that are unique across the released columns, however many distinct values
they hold, and it groups by their released vectors in the k-anonymity and skew checks, so five
records that share one vector and one diagnosis make the release `at_risk`.

MDAV counts rows, not people. With an `entity_column` in the plan, each entity becomes one
point (the mean of its rows), every group holds at least `group_size` distinct entities, and
all the rows of an entity get its group's vector; the k-anonymity, isolated-record and skew
checks then count distinct entities per combination of all released quasi-identifiers (listed
ones and joint columns together), so two people who share a vector and a district are a group of
two, not of ten rows (t-closeness still weighs groups by rows). Without an `entity_column`, the five identical visits of one person are five
records and can form a group on their own.

Limits: the work grows
with the square of the number of rows, so very large tables should be sampled or grouped
first. Someone who knows all but one record of a group can subtract them from the mean to get
the last one, as with `microaggregate`.

### l-diversity enforcement

`enforce_l_diversity: true` runs after k-anonymity. It finds the quasi-identifier groups that
hold fewer than `l_target` distinct values of a sensitive attribute and blanks their
quasi-identifiers, so those rows join the one group whose quasi-identifiers are all blank.
When that group still holds fewer than `l_target` values, the smallest remaining groups join
it until it does, as local suppression does for k. Whole groups are blanked, not single cells,
because a group that keeps some cells keeps the values that identify it. A sensitive attribute
that has fewer than `l_target` values in the whole table cannot reach it, and the protection
check reports it as it did before.

Enforcement costs cells, and the merged group is the least informative one in the release.
Raise `l_target` only as far as the data needs.

### t-closeness

l-diversity counts how many sensitive values a group holds, not how they are spread: a group
that holds 99 percent one diagnosis and 1 percent another passes `l_target: 2` and still tells
an attacker the diagnosis. t-closeness (Li, Li and Venkatasubramanian, 2007) compares each
group's distribution of a sensitive attribute with the whole table's and measures the distance
with the earth mover's distance, from 0 (the same) to 1:
- numbers and dates use the ordered distance, in which moving probability to a neighbouring
  value costs little, so a group of salaries near the table's does not count as far;
- other columns use the equal distance, half the total difference between the shares.

Set `t_target` (a number above 0 and at most 1; null, the default, does not measure it). Then
`result.privacy_after.t_closeness` holds the largest distance of any group per sensitive
attribute, and any attribute above `t_target` is a problem that makes the release `at_risk`.
It is also available alone as `oblitus.evaluation.t_closeness(frame, quasi_identifiers,
sensitive_attributes)`, and `assess_privacy(..., t_target=...)` fills it in the report. Blank
sensitive cells are left out of both distributions. A small `t_target` is hard to meet with
small groups: nothing here enforces it, so the usual fixes are larger groups (a higher
`k_target`), coarser quasi-identifiers or fewer sensitive attributes.

### Skewed groups

l-diversity and t-closeness can both pass while a group still gives a value away: a group of
100 rows with 99 `diabetes` and 1 `healthy` holds two values, and with a loose `t_target` it
passes too. Under anonymization the protection check therefore also looks, per sensitive
attribute, for a quasi-identifier group where one value has
- a share of at least `inference_confidence` (default 0.9), so a guess is right nine times in
  ten, and
- a share at least `inference_gain` (default 0.3) above its share in the whole table, so the
  release taught the attacker something. This stops a table that is 95 percent `healthy` from
  being flagged for a group that is 99 percent `healthy`.

It follows the idea of beta-likeness (Cao and Karras, 2012): bound how much a group's
distribution can raise the attacker's confidence over the base rate. Both limits are plan
fields. A hit is a problem naming the worst group, its value and the shares, and makes the
release `at_risk`. Fix it with `t_target`, coarser quasi-identifiers, or larger groups.
Numeric sensitive columns with more than 50 distinct values are skipped, since no single number has a meaningful share. Text columns are checked however many values they hold. Blank cells are left out.

## Attack-based risk with Anonymeter

The checks above describe how a table is built. `assess_attack_risks` plays the attacker
with [Anonymeter](https://github.com/statice/anonymeter) and reports how often each attack
succeeds, with a confidence interval:

```python
from oblitus.evaluation import assess_attack_risks

report = assess_attack_risks(
    original_pandas,
    released_pandas,
    quasi_identifiers=["age", "cep", "gender"],
    sensitive_attributes=["diagnosis"],
    control=held_out_original_rows,  # optional
    n_attacks=500,
)
for risk in report.risks:
    print(risk.attack, risk.risk, risk.ci_low, risk.ci_high)
```

| Attack | What the attacker does |
|---|---|
| `singling_out` | Writes queries on the quasi-identifier and sensitive columns that each match one original record, using the released table |
| `linkability` | Holds two halves of the quasi-identifiers, as if from two databases, and tries to tell which released records belong together. Needs two quasi-identifiers |
| `inference:<column>` | Knows a person's quasi-identifiers and guesses the sensitive column from the nearest released records, one attack per sensitive attribute |

A risk of 0 means the attacker did no better than guessing. A `control` table, original records
that were not used to build the release, lets Anonymeter separate what the release reveals
from what the population makes easy to guess; without it the risk is the raw success rate.
An attack that cannot run is listed in `report.skipped` with the reason.

The extra is optional: `pip install "oblitus[anonymeter]"`. Without it the function raises
`MissingExtraError`, which names the extra. Anonymeter works on in-memory pandas tables and
runs its attacks in Python, so use a sample for large tables. It adds no protection check:
the numbers inform the contextual assessment and are not a pass or a fail.
`oblitus apply --attack-risks` runs the attacks on the table it wrote and adds them to the
`--report` file under `attack_risks`.

## Plan files

```json
{
  "fidelity": "privacy",
  "protection": "anonymization",
  "actions": [
    {"column": "email", "technique": "suppress", "params": {}, "reason": "email (direct_identifier): removed"},
    {"column": "age", "technique": "bin_numeric", "params": {"width": 10}, "reason": "age (quasi_identifier): ages in decades"}
  ],
  "direct_identifiers": ["email"],
  "row_key": "customer_id",
  "quasi_identifiers": ["age", "cep", "gender"],
  "sensitive_attributes": ["diagnosis"],
  "k_target": 5,
  "l_target": 2,
  "enforce_k_anonymity": true
}
```

Columns without an action are left unchanged. A plan is checked before anything runs.
A missing column, an unknown technique, an unknown parameter, or a linking technique in
an anonymization plan raises `PlanError`.

The table-level fields, all optional, are `k_anonymity_method` (`local_suppression` or
`mondrian`), `range_style` (`range` or `midpoint`), `hierarchies`, `joint_microaggregation`,
`enforce_l_diversity`, `t_target`, `inference_confidence` and `inference_gain`. Each is
described above. Every one defaults to the
behaviour before it existed.

`shuffle_rows` says whether the released rows come back in a fresh random order with a fresh
pandas index. `null` (the default) shuffles under anonymization, so the release cannot be lined
up with the original or another release row by row, and keeps order and index under
pseudonymization. The protection checks still compare the rows before the shuffle.

`row_key` names the column that identifies each row, such as a record ID. `propose`
sets it to the first column found to hold a different value in nearly every row. It does
three things:
- Random numbers for noise are keyed on it, so the same row gets the same noise on
  pandas, Polars, DuckDB and Spark, and on every run with the same keys.
- A DuckDB or Spark table needs it, and its values must be unique, because such a table has
  no fixed row order. pandas and Polars tables without one use the row position.
- It lines up the original and released rows when utility and the value-by-value protection
  checks compare them. The key holds the original values even when the plan replaces
  the column.

`oblitus apply` and `oblitus reveal` take `--row-key COLUMN`, which overrides the
plan's `row_key`.

## Several tables in one call

`Anonymizer.apply_many(frames, plans, vault=None)` takes related tables and one plan per table,
both keyed by table name, and returns one `AnonymizationResult` per table. One plan per table
keeps each plan an ordinary reviewable file, proposed for its own columns. The tables share
what links a person:

- **Pseudonyms** come from the secret key, so the same value gives the same `keyed_hash` or
  consistent `surrogate` in every table.
- **Date offsets** come from the key and the entity column. List `date_shift` with the same
  `by` column (the person's ID), `max_days` and `stream` in each table's plan, and one person's
  dates move by the same number of days in every table, so intervals between events survive.
  Without `stream`, columns with the same name already agree.
- **Tokens** come from the one `vault`, so a value has one token in every table.
- **Anonymization** runs every anonymizing plan of the call under one throwaway key. Category
  codes and random fakes agree between the tables of this release, and cannot be joined to
  any other release.

Every plan is validated against its table before the first table runs, and a refused plan
names its table. Protection is assessed per table, and then on the tables joined. When two or
more anonymizing plans release the same column name with consistent codes (no action, `keep`
or `scramble_categories`), the released tables are joined on it. The check counts entities
there, over the quasi-identifiers of all the joined tables together: a combination shared by
fewer entities than the largest `k_target`, or a group in which a sensitive value passes the
[skewed-group](#skewed-groups) limits, makes every table of that join `at_risk`. The problem
names the other tables and the links. Fix it by dropping the link column from the
release or by releasing the tables in separate calls, so each gets its own codes.

The join follows links through other tables. A bridge table holding an account and a case
code joins a table that has only the account to a table that has only the case code, so the
check joins every connected group of tables through all their links, and counts entities by
the link with the fewest distinct values (the person, when the other links are visits or cases).
A table that holds only some of the entities shrinks the inner join of the whole group, so
every connected subset of two or more tables is joined and checked too, not only the full
group. A group of up to 6 tables is checked in every connected subset; a larger group in every
connected pair and the full join. A problem is reported on the tables of the subset where it
appears.

Two columns with different names are linked too when they code the same entities by how often
they occur: if both hold at least five distinct values, at least 80% of their per-value row
counts agree, and at least half of the values can be paired by a count that no other value has,
the values are paired by count and the tables are joined on that pairing. A column that is also in a named link (a third table copying `account`) is compared as well. The heuristic misses
entities that occur equally often (every account has five rows), tables that were filtered or
sampled so the counts differ, and counts changed by noise. Still not checked: joins through
values you left readable (such as a date), and pseudonymized tables, which join by design. On the command line, `oblitus apply-many` reads a manifest:

```json
{"tables": {
  "customers": {"input": "customers.csv", "output": "out/customers.csv", "plan": "customers_plan.json"},
  "orders": {"input": "orders.csv", "output": "out/orders.csv"}
}}
```

Relative paths are read from the manifest's folder, and a table without a `plan` gets a
proposed one. The exit code is 3 when any table misses its protection target.

## Where keys come from

By default the secret key and the anonymization throwaway key come from `os.urandom`.
`oblitus keygen` and `oblitus.key_generation.generate_key` mix more sources into the key:

| Source | Reads | Needs |
|---|---|---|
| `os` | `os.urandom` | Always included |
| `hardware` | `/dev/hwrng` | A readable Linux hardware generator |
| `microphone` | Low 2 bits of a second of audio samples (thermal noise of the input) | Extra `audio` and the PortAudio library |
| `camera` | Low 2 bits of a few frames (sensor noise) | Extra `camera` and a camera |
| `timing` | Low 2 bits of CPU timing differences | Nothing |
| `beacon` | Latest drand and NIST beacon values over HTTPS, 5 second timeout | Network; opt-in |

Each source's output is tagged with its name and length, concatenated, and extracted with
HKDF-SHA256 into the key. `os.urandom` is always first and cannot be removed, so a weak,
broken or hostile extra source can never make the key weaker than `os.urandom` alone; the
extras only add independence in case the operating system generator is flawed.

Each physical source must deliver at least 64 bytes with a most-common-value min-entropy
estimate of 2 bits per byte or more. Otherwise it is skipped and the report says why. The
report lists bytes, estimated min-entropy and status per source. A missing device is skipped
with a reason, not an error.

A beacon's output is **public**. It adds no secrecy, only that nobody could have predicted the
key before the beacon pulse was published. It is mixed in, never used alone, and the library
never fetches it unless asked.

In Python, `Anonymizer(throwaway_key_source=lambda: generate_key(sources).key)` draws the
anonymization throwaway key from the mix. The library reads no environment variables.

## What keyed noise cannot hide

Noise is computed from the secret key, the seed and the row key, so whoever can recompute it
can subtract it. These limits are not reported per release:

- **A seed without a secret key is public.** Anyone who knows or guesses the seed recomputes
  the noise and gets the original values back exactly. A small seed is found by trying each
  one against a single known record. Releases meant to protect values need a random secret
  key, such as `secrets.token_hex(32)`, kept apart from the release.
- **A guessable key is no key.** A key built from a word, a date or a project name falls to
  the same test: try the candidates against one known value.
- **Re-releasing under the same key shows what changed.** The same row gets the same noise,
  so the difference between two releases of updated data is the exact change in each row.
  Use a new key for each version of the data, or release only once.
- **Releases under different keys or seeds average out.** Each release draws fresh noise, so
  the mean of many releases approaches the true values: the average of 32 releases is about
  six times closer than one. Give each recipient one release.
- **Without a row key, noise follows the row position.** Reordering the input changes which
  row gets which noise, and an attacker who can line up two releases on a public column can
  average them. Set `row_key` when a table is released more than once.
- **The engine sees the derived key.** On DuckDB and Spark the random numbers are computed in SQL, so the
  per-column key derived from the secret key appears in the query text and in any query log
  or profile. It reveals that column's noise, not the secret key. Do not run releases on a
  connection whose query log others can read.

## What the checks cannot catch

The protection report checks the released table against fixed rules. A release can pass them
and still let someone who knows their own records follow those records:

- **Small keys and combinations.** The report flags an untreated text or whole-number column
  with at least 20 distinct values that group rows, and pairs of such columns. A key with fewer
  values, a key made of three or more columns, a pair of columns that each have fewer than 20
  values, or a decimal key with a few fractional cells passes. List every column in the plan,
  and drop the ones the recipient does not need.
- **Consistent pseudonyms follow the value's frequency.** A keyed hash or a consistent
  surrogate gives the most common name the most common token. Someone who knows how common
  each name is can decode the frequent ones: in a measured test, ranking 40 first names by
  public frequency decoded about 60% of rows of a pseudonymized column, against about 20% for
  guessing the most common name. Anonymization replaces such columns with fresh codes.
- **Rounded columns join two releases.** Codes and fakes are fresh in every run, but a column
  that is only rounded or microaggregated gives the same values in two releases. Two releases
  of different columns of one table can be joined on them; in a measured test, a rounded
  amount and an age in groups of five joined 48% of the rows of a 1,500-row table correctly.
  Release one table, or give the releases no generalized column in common.
- **Numeric codes are found by their spread.** Unique whole numbers spread evenly over the
  range they cover within each digit length, or nearly all of one length with at least 6
  digits, are taken for codes and replaced. Codes from a few narrow ranges (one per branch,
  say) with fewer than 6 digits pass as numbers. The other way round, an unnamed measure whose
  values all have the same number of digits, or any unnamed unique measure in a table of about
  100 rows, where the test cannot tell them apart, is replaced too, which costs utility, not
  privacy. Naming the column
  after a quantity (`amount`, `valor`, `saldo`, `population` and similar) keeps it a number.
- **Group sizes and own values survive fresh codes.** Replacing an account with a fresh code
  hides the value, not the group. Someone who knows how many rows they have, or a few of their
  own amounts, can find their code and read the rest of their rows. Perturb or generalize the
  numeric columns as well when that matters.
- **Business figures need an entity.** Customer counts per merchant, revenue shares and trends
  are computed from released rows. The dominance check only sees them when the plan names the
  `entity_column`; see [business-confidentiality.md](business-confidentiality.md).

## Large tables

`Anonymizer` takes pandas, Polars, DuckDB and Spark tables and returns the kind it was given.
A Spark DataFrame stays a Spark query until a result is read: keyed noise is computed with
`sha2` in Spark SQL and gives the same numbers as on every other engine, and lookups are
joined on your own session. The Spark engine is tested on `sqlframe`, which has the PySpark
API on top of DuckDB, and on a local PySpark session in CI. It has not been run on a Spark
cluster. Spark column names cannot
contain dots or backticks.
- k-anonymity, l-diversity, local suppression, record isolation and fingerprint columns
  are computed on the whole table by the engine that holds it. Mondrian, joint
  microaggregation, l-diversity enforcement and t-closeness collect the distinct groups (or,
  for microaggregation, the chosen columns) into memory, then write the result back on the
  table's engine.
- Utility, disclosure, dominance, surviving identifiers and noisy-value linkage compare each
  released value with its original. They run on the same at most 200,000 rows of both tables,
  picked by a keyed hash of the row key, so the sample is the same on every run and
  engine. Smaller tables are compared whole.
- Dominance sums only the sampled rows of each contributor, so for a table above the limit
  the shares are estimates.
- A DuckDB `TIMESTAMP WITH TIME ZONE` column is read in the connection's time zone, so date
  techniques see local times. Set the connection's `TimeZone` before applying a plan when the
  release must use UTC.
- A Spark `TIMESTAMP` is read back in the session time zone, so set `spark.sql.session.timeZone`
  before applying a plan when the release must use UTC.

## Files larger than memory

`oblitus apply INPUT --plan PLAN --chunk-rows N -o OUTPUT`, or `Anonymizer.apply_file`, reads a
CSV or Parquet file N rows at a time, runs the plan's actions on each chunk and appends the
released rows to the output. Only one chunk is in memory at once.

A technique can run in chunks only when a row's result does not depend on the other rows. These
run in chunks and release the same values as an in-memory run on a file that fits:
`keep`, `redact`, `suppress`, `partial_mask`, `redact_entities`, `keyed_hash`,
`scramble_characters`, `argon2_hash`, `bcrypt_hash`, `encrypt`, `format_preserving_encrypt`,
`tokenize`, `round_numeric`, `truncate_date`, `truncate_string`, `generalize_hierarchy`,
`laplace_noise`, `multiplicative_noise` and `date_shift`. Keyed noise streams because each
row's numbers come from its row key. `bin_numeric` runs with a whole-number `width`, and
`top_bottom_code` with both `lower` and `upper` fixed.

These are refused, before any row is read, with a `PlanError` that lists every reason:
- techniques that read the whole column: `normalize`, `bin_numeric` by quantiles or bins,
  `top_bottom_code` by quantiles, `gaussian_noise` (it rescales to the column's mean and
  deviation), `shuffle`, `rank_swap`, `microaggregate`, `scramble_categories`, `suppress_rare`,
  `randomized_response` and `surrogate`;
- `truncate_date` and `date_shift` on a text column, because the date layout is guessed from
  each chunk's values; convert the column to a date type first;
- the plan's `enforce_k_anonymity`, `enforce_l_diversity` and `joint_microaggregation`;
- a plan without a `row_key`. The key must be unique in the whole file, which a pass over it
  checks first;
- an anonymization plan whose `shuffle_rows` is not `false`, because a chunked release keeps
  the input row order. A release that lines up with the original row by row is a weakness
  `shuffle_rows` normally removes, so decide on it yourself.

None of these has a first pass that collects its statistics. Run them in memory, or on a
DuckDB table, where the engine does the work.

The protection level is computed, not assumed. After the last chunk, the whole-table checks
(k-anonymity, l-diversity, t-closeness, record isolation, fingerprint columns) read the output
file, and the input file for the "before" figures, lazily and with the streaming engine, so each
pass touches only the columns it needs. The value-by-value checks use the same keyed sample of at
most 200,000 rows as an in-memory run. The reported level, problems and figures equal an
in-memory run's. A run that fails part-way removes its partly written output.

CSV chunks are split at line breaks outside quotes. A stray quote character inside an unquoted
field makes the chunk's row count differ from the file's, and the run stops with a message. The
first chunk fixes the output's column types, and a later chunk that cannot be cast to them stops
the run.
