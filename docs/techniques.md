# Techniques

`oblitus techniques` prints this list from the code.

| Technique | Family | Reversible | Keeps a link | Effect on statistics | Column kinds | Parameters | What it does |
|---|---|---|---|---|---|---|---|
| `keep` | keep | no | no | preserves relationships | any | — | Leave unchanged. |
| `redact` | masking | no | no | distorts | any | `replacement` | Replace with a constant. |
| `suppress` | masking | no | no | distorts | any | — | Replace with missing. |
| `partial_mask` | masking | no | no | distorts | category, integer, text | `keep_first`, `keep_last`, `mask_char` | Hide all but the first or last characters, keeping separators. |
| `redact_entities` | masking | no | no | distorts | category, text | `categories` | Replace emails, documents and phones inside free text with labels. Text is normalized first (zero-width characters, look-alike letters and digits, `[at]` and `[dot]`, document numbers split by spaces or punctuation, phones split by number words). Names stay. |
| `keyed_hash` | pseudonymization | no | yes | preserves relationships | any | `length`, `prefix`, `algorithm` | Replace with a keyed HMAC digest (SHA-2, SHA-3 or BLAKE2); consistent per key. `length` is 16 to 128 hex characters. |
| `scramble_categories` | pseudonymization | no | only with `keyed: true` | preserves relationships | any | `prefix`, `min_count`, `min_entities`, `by`, `keyed` | Replace labels with opaque codes in random order; labels with fewer than `min_count` rows, or held by fewer than `min_entities` distinct values of the `by` column, become `<prefix>-OTHER`, which grows with the smallest segments until it meets the same minimums (blank if it cannot). |
| `scramble_characters` | pseudonymization | no | yes | preserves relationships | category, integer, text | — | Replace digits with digits and letters with letters, keeping the format. Uses FF1 per character class, so distinct values stay distinct; runs shorter than 6 digits or 5 letters fall back to a keyed hash. |
| `surrogate` | pseudonymization | no | yes, unless `consistent: false` | preserves relationships | category, integer, text | `category`, `unique`, `consistent` | Replace with a realistic fake; consistent per value, or random per cell. A fake never equals a real value of the column. |
| `argon2_hash` | pseudonymization | no | yes | preserves relationships | any | `time_cost`, `memory_cost`, `parallelism`, `hash_length`, `deterministic` | Replace with a slow, memory-hard Argon2id hash; a keyed salt keeps joins. |
| `bcrypt_hash` | pseudonymization | no | yes | preserves relationships | any | `rounds`, `deterministic` | Replace with a slow bcrypt hash; a keyed salt keeps joins. |
| `encrypt` | encryption | yes | yes | preserves relationships | any | `deterministic` | Reversible AES-SIV token (deterministic) or AES-GCM token (randomized). |
| `format_preserving_encrypt` | encryption | yes | yes | preserves relationships | category, integer, text | `alphabet`, `tweak` | Reversible NIST FF1 encryption that keeps length, format and separators. |
| `bin_numeric` | generalization | no | no | distorts | float, integer | `bins`, `width`, `strategy` | Replace numbers with interval labels. |
| `round_numeric` | generalization | no | no | preserves distribution | float, integer | `decimals`, `base` | Round to decimals or to a multiple of a base. |
| `truncate_date` | generalization | no | no | distorts | datetime | `unit` | Cut dates to the start of their year, month, week or day. |
| `truncate_string` | generalization | no | no | distorts | category, integer, text | `keep`, `pad_char` | Keep only the first characters, such as a postal code's region. |
| `generalize_hierarchy` | generalization | no | no | distorts | boolean, category, integer, text | `hierarchy`, `level` | Replace each value with its stand-in at a level of a hierarchy, such as a postal code's region. See [Generalization hierarchies](#generalization-hierarchies). |
| `top_bottom_code` | generalization | no | no | preserves distribution | float, integer | `lower_quantile`, `upper_quantile`, `lower`, `upper` | Clip outliers to quantile or fixed limits. |
| `suppress_rare` | generalization | no | no | distorts | boolean, category, integer, text | `min_count`, `replacement` | Replace categories seen fewer than a minimum number of times. |
| `gaussian_noise` | noise | no | no | preserves distribution | float, integer | `relative_scale`, `scale`, `preserve_variance` | Add normal noise, rescaled to keep the mean and standard deviation. |
| `laplace_noise` | noise | no | no | distorts | float, integer | `epsilon`, `lower`, `upper` | Clip and add Laplace noise; local differential privacy with fixed bounds. `lower` and `upper` are required. |
| `normalize` | rescaling | no | no | preserves shape | float, integer | `method`, `by`, `order_by`, `allow_shape_change` | Recentre and rescale to hide level and units. `robust` (default), `zscore` and `minmax` keep the shape; `index`, `rank`, `log_ratio`, `growth` and any `by` need `allow_shape_change`. |
| `multiplicative_noise` | noise | no | no | preserves distribution | float, integer | `relative_scale` | Multiply by a random factor near 1; suits amounts. |
| `date_shift` | noise | no | no | preserves distribution | datetime | `max_days`, `by` | Move each date by a random number of days; with `by`, every row of one entity moves by the same offset. |
| `shuffle` | perturbation | no | no | preserves distribution | any | — | Reorder values among rows; exact distribution, no link to other columns. |
| `rank_swap` | perturbation | no | no | preserves distribution | datetime, float, integer | `max_distance_percent` | Swap values with near neighbours in rank; exact distribution. |
| `microaggregate` | perturbation | no | no | preserves distribution | float, integer | `group_size` | Replace values with the mean of groups of at least k; exact mean, so someone who knows the other values of a group can subtract them to get the last one. |
| `randomized_response` | perturbation | no | no | distorts | boolean, category | `keep_probability`, `by` | Randomly replace some categories; deniable, local differential privacy. With `by`, all rows of one entity share one draw, so averaging them reveals nothing. |

Numeric techniques compute in 64-bit floating point, so integers above 2^53 (about
9 × 10^15) lose precision when a numeric technique transforms them. Use a key technique
such as `keyed_hash` for very large identifiers. Columns no action changes, and row keys,
come back exactly. For pandas tables, untouched columns come back as the user's own
columns, with their dtype and index.

What the effects mean:

- **preserves relationships**: distinct counts, joins and links to other columns survive.
- **preserves distribution**: the column's own distribution survives, exactly or
  approximately, but links to other columns may weaken.
- **preserves shape**: the values are shifted and scaled, so the distribution's shape,
  the ranking and the correlations survive, but the units and level do not.
- **distorts**: the distribution changes, and analysis has to account for it.

**Keeps a link** means the output can be tied back to the person by the key holder, or by
anyone testing a guess against a published hash. Anonymization plans refuse those
techniques.

## Generalization hierarchies

A hierarchy says how a value becomes coarser, level by level. Level 0 is the value, level 1 the
first stand-in, and the level after the last one is `*`. Write it in one of three forms:

```json
{"levels": [{"Santos": "Baixada", "Bauru": "Interior"}, {"Santos": "SP", "Bauru": "SP"}]}
{"prefix_lengths": [5, 3], "digits_only": true}
{"preset": "cep"}
```

- `levels` holds one mapping per level, each from the original value to its stand-in at that
  level. A value a mapping does not list stays as it is at that level.
- `prefix_lengths` keeps the first characters and stars the rest, from specific to general and
  strictly decreasing. `digits_only` drops separators first.
- `{"preset": "cep"}` is the Brazilian postal code form of the second: `01310-100` becomes
  `01310***`, `013*****`, `01******` and `0*******` by keeping 5, 3, 2 and 1 digits. It needs
  no external data, so it names a delivery area and a region by prefix, not a city or a state.

The `generalize_hierarchy` technique applies one level to a column:
`{"technique": "generalize_hierarchy", "params": {"hierarchy": {"preset": "cep"}, "level": 2}}`.
The plan's `hierarchies` field maps a column to the same definition, for Mondrian to describe a
group of categories (see [protection.md](protection.md#mondrian-generalization)).

## Joint microaggregation

`microaggregate` protects one column at a time. A record's vector over several columns can
still be rare, so the plan's `joint_microaggregation` list groups records by their distance
over several numeric columns at once (the MDAV method) and gives each column its group's mean.
It is a plan step, because a technique sees one column. See
[protection.md](protection.md#joint-microaggregation).

## One-way or reversible

- **One-way** (`redact`, `keyed_hash`, `argon2_hash`, `bcrypt_hash`, `surrogate`, ...):
  nobody can recover the value.
  - Pick Argon2id or bcrypt when values are easy to guess, such as document numbers. Their
    cost slows a brute-force search even for someone who holds the key.
  - They run once per distinct value. On one modern CPU core a value costs 10 to 20 ms with
    the Argon2id defaults, and 40 to 60 ms with bcrypt at 10 rounds.
- **Reversible** (`encrypt`, `format_preserving_encrypt`): `reveal` recovers the value.
  - Use it when an authorized team must get back to the person, for example to act on a
    fraud alert.
  - Deterministic `encrypt` and FF1 give equal outputs for equal inputs, so joins survive.
    `encrypt` with `"deterministic": false` gives a different token every time.
  - FF1 needs at least 6 digits, or 4 characters with the `alphanumeric` alphabet.

`laplace_noise` satisfies ε-local differential privacy only with `lower` and `upper` fixed
in advance, without looking at the data, so both are required. Taking them from the
column's minimum and maximum would leak those two numbers.

## Your own techniques

1. Write a function `(table, column, context, **params)` that returns a Narwhals
   expression for the column's new values, such as
   `nw.col(column).str.to_uppercase()`. Build it from expressions, and it runs on pandas,
   Polars and DuckDB alike. `Table` (in `oblitus.table`) provides:
   - statistics;
   - keyed random numbers per row (`table.uniforms(context.stream_key(column, "mine"))`);
   - lookups of values computed in Python (`value_mapping.lookup_distinct`).
2. Wrap it in a `Technique`. If it is reversible, set `inverse=`. If its output can be tied
   back to a person, set `linkable=True`.
3. Pass it as `Anonymizer(extra_techniques=[...])`. Plans can then name it.
