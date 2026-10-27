# Techniques

`anonymize techniques` prints this list from the code.

| Technique | Family | Reversible | Keeps a link | Effect on statistics | Column kinds | Parameters | What it does |
|---|---|---|---|---|---|---|---|
| `keep` | keep | no | no | preserves relationships | any | — | Leave unchanged. |
| `redact` | masking | no | no | distorts | any | `replacement` | Replace with a constant. |
| `suppress` | masking | no | no | distorts | any | — | Replace with missing. |
| `partial_mask` | masking | no | no | distorts | category, integer, text | `keep_first`, `keep_last`, `mask_char` | Hide all but the first or last characters, keeping separators. |
| `redact_entities` | masking | no | no | distorts | category, text | `categories` | Replace emails, documents and phones inside free text with labels. |
| `keyed_hash` | pseudonymization | no | yes | preserves relationships | any | `length`, `prefix`, `algorithm` | Replace with a keyed HMAC digest (SHA-2, SHA-3 or BLAKE2); consistent per key. |
| `scramble_characters` | pseudonymization | no | yes | preserves relationships | category, integer, text | — | Replace digits with digits and letters with letters, keeping the format. |
| `surrogate` | pseudonymization | no | yes, unless `consistent: false` | preserves relationships | category, integer, text | `category`, `unique`, `consistent` | Replace with a realistic fake; consistent per value, or random per cell. |
| `argon2_hash` | pseudonymization | no | yes | preserves relationships | any | `time_cost`, `memory_cost`, `parallelism`, `hash_length`, `deterministic` | Replace with a slow, memory-hard Argon2id hash; a keyed salt keeps joins. |
| `bcrypt_hash` | pseudonymization | no | yes | preserves relationships | any | `rounds`, `deterministic` | Replace with a slow bcrypt hash; a keyed salt keeps joins. |
| `encrypt` | encryption | yes | yes | preserves relationships | any | `deterministic` | Reversible AES-SIV token (deterministic) or AES-GCM token (randomized). |
| `format_preserving_encrypt` | encryption | yes | yes | preserves relationships | category, integer, text | `alphabet`, `tweak` | Reversible NIST FF1 encryption that keeps length, format and separators. |
| `bin_numeric` | generalization | no | no | distorts | float, integer | `bins`, `width`, `strategy` | Replace numbers with interval labels. |
| `round_numeric` | generalization | no | no | preserves distribution | float, integer | `decimals`, `base` | Round to decimals or to a multiple of a base. |
| `truncate_date` | generalization | no | no | distorts | datetime | `unit` | Cut dates to the start of their year, month, week or day. |
| `truncate_string` | generalization | no | no | distorts | category, integer, text | `keep`, `pad_char` | Keep only the first characters, such as a postal code's region. |
| `top_bottom_code` | generalization | no | no | preserves distribution | float, integer | `lower_quantile`, `upper_quantile`, `lower`, `upper` | Clip outliers to quantile or fixed limits. |
| `suppress_rare` | generalization | no | no | distorts | boolean, category, integer, text | `min_count`, `replacement` | Replace categories seen fewer than a minimum number of times. |
| `gaussian_noise` | noise | no | no | preserves distribution | float, integer | `relative_scale`, `scale`, `preserve_variance` | Add normal noise, rescaled to keep the mean and standard deviation. |
| `laplace_noise` | noise | no | no | distorts | float, integer | `epsilon`, `lower`, `upper` | Clip and add Laplace noise; local differential privacy with fixed bounds. |
| `multiplicative_noise` | noise | no | no | preserves distribution | float, integer | `relative_scale` | Multiply by a random factor near 1; suits amounts. |
| `date_shift` | noise | no | no | preserves distribution | datetime | `max_days` | Move each date by a random number of days. |
| `shuffle` | perturbation | no | no | preserves distribution | any | — | Reorder values among rows; exact distribution, no link to other columns. |
| `rank_swap` | perturbation | no | no | preserves distribution | datetime, float, integer | `max_distance_percent` | Swap values with near neighbours in rank; exact distribution. |
| `microaggregate` | perturbation | no | no | preserves distribution | float, integer | `group_size` | Replace values with the mean of groups of at least k; exact mean. |
| `randomized_response` | perturbation | no | no | distorts | boolean, category | `keep_probability` | Randomly replace some categories; deniable, local differential privacy. |

What the effects mean:

- **preserves relationships**: distinct counts, joins and links to other columns survive.
- **preserves distribution**: the column's own distribution survives, exactly or
  approximately, but links to other columns may weaken.
- **distorts**: the distribution changes, and analysis has to account for it.

**Keeps a link** means the output can be tied back to the person by the key holder, or by
anyone testing a guess against a published hash. Anonymization plans refuse those
techniques.

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

`laplace_noise` satisfies ε-local differential privacy only when you pass `lower` and
`upper`. Without them it uses the column's minimum and maximum, and those two numbers
leak.

## Your own techniques

1. Write a function `(values, context, **params) -> Series`.
2. Wrap it in a `Technique`. If it is reversible, set `inverse=`. If its output can be tied
   back to a person, set `linkable=True`.
3. Pass it as `Anonymizer(extra_techniques=[...])`. Plans can then name it.
