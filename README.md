# data-anonymizer

A Python library and command-line tool for tables that contain personal data. It:

1. **detects** which columns hold personal or sensitive data, from column names and values
   (emails, phones, names, CPF/CNPJ, SSN, card numbers, IBAN, IPs, addresses, birth dates,
   free text with personal data embedded, and more);
2. **proposes** a plan, a JSON file you can review and edit, that assigns one of 26
   techniques to each flagged column;
3. **applies** the plan and reports k-anonymity, l-diversity and statistical drift;
4. **reveals** the original values of reversibly encrypted columns, for whoever holds the key;
5. **generates** synthetic tables from a schema you write, or from a sample table.

Every plan and every synthesis run uses one of two fidelity modes. **`statistical`** keeps
distributions, correlations, distinct counts and joins. **`privacy`** generalizes, suppresses
and randomizes until records are hard to single out.

## Features and algorithms

| Area | What is implemented |
|---|---|
| Detection | Regular expressions with checksum validation: CPF and CNPJ check digits, Luhn for card numbers, ISO 13616 mod-97 for IBAN, issued-range check for SSN. Column-name hints in English and Portuguese (camelCase, accents, `_id` suffixes). Free-text scanning for embedded personal data. Uniqueness-based record-ID detection. Storage-kind inference (integer, float, boolean, category, datetime, text). A confidence score and evidence for every finding |
| Masking, one-way | Redaction, suppression, partial masking that keeps separators, entity redaction inside free text |
| Hashing, one-way | HMAC with SHA-256, SHA-512, SHA3-256, SHA3-512, BLAKE2b or BLAKE2s. **Argon2id** and **bcrypt**, with a keyed deterministic salt (joins survive) or a random salt (unlinkable) |
| Pseudonymization | Keyed format-preserving character scrambling. Realistic surrogates (names, emails, phones, valid CPF/CNPJ, SSN, cards, IBAN, IPs, addresses, URLs), consistent per value and collision-free |
| Encryption, reversible | **AES-SIV** deterministic encryption (RFC 5297). **AES-256-GCM** randomized encryption. **FF1** format-preserving encryption (NIST SP 800-38G, checked against the NIST sample vectors). Keys derived from one secret with **HKDF-SHA256** |
| Generalization | Fixed-width, equal-width and quantile binning. Rounding to decimals or a base. Date truncation to year, month, week or day. String truncation. Top and bottom coding. Rare-category suppression |
| Noise | Gaussian noise rescaled to keep mean and variance. **Laplace mechanism** (ε-local differential privacy with fixed bounds). Multiplicative noise. Random date shifting |
| Perturbation | Shuffling, rank swapping, univariate microaggregation, **randomized response** (local differential privacy) |
| Privacy models | k-anonymity and l-diversity measurement. k-anonymity enforcement by greedy local suppression |
| Utility metrics | Relative change of mean and standard deviation, Kolmogorov–Smirnov statistic, total variation distance, maximum change in pairwise correlation |
| Synthesis | Schema-driven generation: bounded normal, uniform, weighted choices, date ranges, templates, unique keys, null shares. Schema inference from a table. **Gaussian copula** that keeps correlations across numeric, date and category columns. Privacy-first independent marginals with rare categories removed |
| Interfaces | Python API (`Anonymizer`), editable JSON plans and schemas, `anonymize` CLI, CSV and Parquet files, custom techniques |

### Roadmap

Planned, not yet implemented:

- **Tokenization with a vault**: random tokens with the mapping kept in a separate
  store, so a token reveals nothing even when the key leaks.
- **Entity-consistent transformations**: shift all dates of one person by the same
  offset, and keep pseudonyms consistent across several tables in one call.
- **Stronger k-anonymity**: Mondrian multidimensional generalization, generalization
  hierarchies (postal code → region → state), multivariate microaggregation (MDAV), and
  t-closeness.
- **Differentially private synthesis**: marginal-based generators (MST, AIM, PrivBayes) with
  an ε budget, and DP aggregate statistics.
- **Synthetic-data privacy checks**: distance to the closest real record and
  membership-inference tests.
- **Detection**: named-entity recognition for names and addresses inside free text (an
  optional spaCy or Presidio extra), names detected from their values, and more national
  identifiers (RG, PIS, CNH, NIF, NINO, Aadhaar).
- **Scale**: Polars and chunked or streaming processing for files larger than memory.

## Quick start

```bash
git clone https://github.com/rmonteiro-pereira/data-anonymizer.git
cd data-anonymizer
uv sync
export ANONYMIZER_SECRET_KEY="$(openssl rand -hex 32)"

uv run anonymize scan customers.csv
uv run anonymize propose customers.csv --fidelity statistical -o plan.json
# review and edit plan.json, then:
uv run anonymize apply customers.csv --plan plan.json -o customers_anon.csv --report report.json
```

The same flow in Python:

```python
from pathlib import Path

import pandas as pd
from data_anonymizer import Anonymizer, Fidelity

customers = pd.read_csv("customers.csv")
anonymizer = Anonymizer(secret_key="a long secret", seed=42, locale="pt_BR")

for finding in anonymizer.scan(customers):
    print(finding.column, finding.category, finding.sensitivity, finding.confidence)

plan = anonymizer.propose(customers, fidelity=Fidelity.PRIVACY)
plan.save(Path("plan.json"))  # optional: review it as JSON

result = anonymizer.apply(customers, plan)
result.data  # the anonymized DataFrame
result.privacy_after.k_anonymity  # smallest group of rows sharing quasi-identifiers
result.utility.max_correlation_change  # how much numeric correlations moved
```

## Configuration

| Variable | Required | Default | What it does |
|---|---|---|---|
| `ANONYMIZER_SECRET_KEY` | for `reveal` | random per run | Keys every pseudonym and keyed salt, and is the master secret for the encryption keys. With the same key, the same value always gets the same pseudonym, so tables anonymized separately still join. `reveal` refuses to run without it. `apply` warns and uses a random key, so pseudonyms change on every run and encrypted columns can never be revealed. |

In Python, pass `secret_key=` to `Anonymizer`; the library never reads the environment.
Treat the key like a password. Anyone who holds it can decrypt the reversible columns,
and can test whether a guessed real value produces a given pseudonym.

## Usage

### Command line

All commands take `--seed N` (reproducible noise and synthesis) and `--locale` (Faker
locale for fake values, such as `pt_BR`; default `en_US`). Input and output files can be
`.csv` or `.parquet`. Parquet needs `uv sync --extra parquet`.

| Command | What it does |
|---|---|
| `anonymize scan INPUT [--sample-size N]` | Prints each column's category, sensitivity, confidence and evidence |
| `anonymize propose INPUT [--fidelity F] [-o PLAN]` | Writes a plan as JSON (stdout without `-o`) |
| `anonymize apply INPUT -o OUTPUT [--plan PLAN] [--fidelity F] [--report REPORT]` | Anonymizes. Without `--plan` it proposes one first. `--report` writes privacy and utility metrics as JSON |
| `anonymize reveal INPUT --plan PLAN -o OUTPUT` | Decrypts the columns the plan encrypted (`encrypt`, `format_preserving_encrypt`) and leaves one-way columns as they are |
| `anonymize infer-schema INPUT [--fidelity F] [-o SCHEMA]` | Learns a generation schema from a table |
| `anonymize synthesize -o OUTPUT --rows N [--schema SCHEMA] [--reference INPUT] [--fidelity F]` | Generates a synthetic table |
| `anonymize techniques` | Lists every technique with its parameters |

`F` is `statistical` (the default) or `privacy`. `-o` is short for `--output`. Errors exit
with code 2.

### Detection

Each column gets a `DataCategory` and a `Sensitivity`:

| Sensitivity | Meaning | Categories |
|---|---|---|
| direct identifier | Points at one person on its own | email, phone, person_name, cpf, cnpj, ssn, credit_card, iban, ip_address, street_address, record_id, free_text_with_pii |
| quasi-identifier | Identifies in combination | url, postal_code, city, date_of_birth, age, gender, geo_coordinate |
| sensitive | What an attacker wants to learn | income, health |
| non-sensitive | Nothing found | generic_numeric, generic_category, generic_datetime, generic_text |

Values are checked against patterns. Where a checksum exists it is verified too:
CPF and CNPJ check digits, the Luhn check on card numbers, and IBAN mod-97. Column names
are matched in English and Portuguese, including camelCase and accented names. A
value match scores higher than a name match. When the name and the values disagree,
confidence drops, and the plan marks that column "please review".

### Fidelity modes and the default proposal

| Category | `statistical` | `privacy` |
|---|---|---|
| email, phone, name, documents, card, IBAN, IP, address, URL | `surrogate`: realistic fake, consistent per value | `redact` |
| record_id | `keyed_hash` | `suppress` (use `keyed_hash` if rows must stay linkable) |
| free text with personal data | `redact_entities` | `redact_entities` |
| postal_code | `truncate_string` keep 3 | `truncate_string` keep 2 |
| city | `keep` | `suppress_rare` below 10 rows |
| date_of_birth | `date_shift` ±30 days | `truncate_date` to year |
| age | `microaggregate` groups of 5 | `bin_numeric` decades |
| gender | `keep` | `keep` |
| geo_coordinate | `gaussian_noise` about 1 km | `round_numeric` 1 decimal |
| income | `rank_swap` within 5% | `bin_numeric` quintiles |
| health | `keep` | `randomized_response` p = 0.75 |

Privacy plans also set `enforce_k_anonymity`. After the column techniques run,
quasi-identifier cells are blanked in rows whose combination is shared by fewer than
`k_target` (default 5) rows. A single row that is unique on every quasi-identifier cannot
be hidden by blanking alone, and the privacy report shows it.

Statistical outputs are **pseudonymous, not anonymous**. Joins survive because the same
key maps the same person to the same pseudonym. Under GDPR and LGPD, pseudonymous data is
still personal data.

### Techniques

| Technique | Family | Reversible | Effect on statistics | Column kinds | Parameters | What it does |
|---|---|---|---|---|---|---|
| `keep` | keep | no | preserves relationships | any | — | Leave unchanged. |
| `redact` | masking | no | distorts | any | `replacement` | Replace with a constant. |
| `suppress` | masking | no | distorts | any | — | Replace with missing. |
| `partial_mask` | masking | no | distorts | category, integer, text | `keep_first`, `keep_last`, `mask_char` | Hide all but the first or last characters, keeping separators. |
| `redact_entities` | masking | no | distorts | category, text | `categories` | Replace emails, documents and phones inside free text with labels. |
| `keyed_hash` | pseudonymization | no | preserves relationships | any | `length`, `prefix`, `algorithm` | Replace with an HMAC-SHA256 digest; consistent under one secret key. |
| `scramble_characters` | pseudonymization | no | preserves relationships | category, integer, text | — | Replace digits with digits and letters with letters, keeping the format. |
| `surrogate` | pseudonymization | no | preserves relationships | category, integer, text | `category`, `unique` | Replace with a realistic fake value, the same fake for the same real value. |
| `argon2_hash` | pseudonymization | no | preserves relationships | any | `time_cost`, `memory_cost`, `parallelism`, `hash_length`, `deterministic` | Replace with a slow, memory-hard Argon2id hash; keyed salt keeps joins. |
| `bcrypt_hash` | pseudonymization | no | preserves relationships | any | `rounds`, `deterministic` | Replace with a slow bcrypt hash; keyed salt keeps joins. |
| `encrypt` | encryption | yes | preserves relationships | any | `deterministic` | Reversible AES-SIV token (deterministic) or AES-GCM token (randomized). |
| `format_preserving_encrypt` | encryption | yes | preserves relationships | category, integer, text | `alphabet`, `tweak` | Reversible NIST FF1 encryption that keeps length, format and separators. |
| `bin_numeric` | generalization | no | distorts | float, integer | `bins`, `width`, `strategy` | Replace numbers with interval labels. |
| `round_numeric` | generalization | no | preserves distribution | float, integer | `decimals`, `base` | Round to decimals or to a multiple of a base. |
| `truncate_date` | generalization | no | distorts | datetime | `unit` | Cut dates to the start of their year, month, week or day. |
| `truncate_string` | generalization | no | distorts | category, integer, text | `keep`, `pad_char` | Keep only the first characters, such as a postal code's region. |
| `top_bottom_code` | generalization | no | preserves distribution | float, integer | `lower_quantile`, `upper_quantile`, `lower`, `upper` | Clip outliers to quantile or fixed limits. |
| `suppress_rare` | generalization | no | distorts | boolean, category, integer, text | `min_count`, `replacement` | Replace categories seen fewer than a minimum number of times. |
| `gaussian_noise` | noise | no | preserves distribution | float, integer | `relative_scale`, `scale`, `preserve_variance` | Add normal noise, rescaled to keep the mean and standard deviation. |
| `laplace_noise` | noise | no | distorts | float, integer | `epsilon`, `lower`, `upper` | Clip and add Laplace noise; local differential privacy with fixed bounds. |
| `multiplicative_noise` | noise | no | preserves distribution | float, integer | `relative_scale` | Multiply by a random factor near 1; suits amounts. |
| `date_shift` | noise | no | preserves distribution | datetime | `max_days` | Move each date by a random number of days. |
| `shuffle` | perturbation | no | preserves distribution | any | — | Reorder values among rows; exact distribution, no link to other columns. |
| `rank_swap` | perturbation | no | preserves distribution | datetime, float, integer | `max_distance_percent` | Swap values with near neighbours in rank; exact distribution. |
| `microaggregate` | perturbation | no | preserves distribution | float, integer | `group_size` | Replace values with the mean of groups of at least k; exact mean. |
| `randomized_response` | perturbation | no | distorts | boolean, category | `keep_probability` | Randomly replace some categories; deniable, local differential privacy. |

What the effects mean:

- **preserves relationships**: distinct counts, joins and links to other columns survive.
- **preserves distribution**: the column's own distribution survives, exactly or
  approximately, but links to other columns may weaken.
- **distorts**: the distribution changes, and analysis has to account for it.

Choosing between one-way and reversible protection:

- **One-way** (`redact`, `keyed_hash`, `argon2_hash`, `bcrypt_hash`, `surrogate`, ...):
  nobody can recover the value. Pick Argon2id or bcrypt when values are easy to guess,
  such as document numbers. Their cost slows a brute-force search even for someone who
  holds the key. They run once per distinct value. On one modern CPU core a value costs
  10 to 20 ms with the Argon2id defaults, and 40 to 60 ms with bcrypt at 10 rounds.
- **Reversible** (`encrypt`, `format_preserving_encrypt`): `reveal` recovers the value.
  Use it when an authorized team must get back to the person, for example to act on a
  fraud alert. Deterministic `encrypt` and FF1 give equal outputs for equal inputs, so joins
  survive. `encrypt` with `"deterministic": false` gives a different token every time.
  FF1 needs at least 6 digits, or 4 characters with the `alphanumeric` alphabet.

`laplace_noise` satisfies ε-local differential privacy only when you pass `lower` and
`upper`. Without them it uses the column's minimum and maximum, and those two numbers
leak.

To add your own technique, write a function `(values, context, **params) -> Series`, wrap
it in a `Technique`, and pass it as `Anonymizer(extra_techniques=[...])`. Plans can then
name it.

### A plan file

```json
{
  "fidelity": "privacy",
  "actions": [
    {"column": "email", "technique": "redact", "params": {}, "reason": "email (direct_identifier)"},
    {"column": "age", "technique": "bin_numeric", "params": {"width": 10}, "reason": "age (quasi_identifier): ages in decades"}
  ],
  "quasi_identifiers": ["age", "cep", "gender"],
  "sensitive_attributes": ["diagnosis"],
  "k_target": 5,
  "enforce_k_anonymity": true
}
```

Columns without an action are left unchanged. A plan is checked before anything runs.
A missing column, an unknown technique or an unknown parameter raises `PlanError`.

### Synthetic data

From a schema you write:

```python
from pathlib import Path

from data_anonymizer import Anonymizer, DatasetSchema

schema = DatasetSchema.load(Path("examples/customers_schema.json"))
synthetic = Anonymizer(seed=1, locale="pt_BR").synthesize(10_000, schema=schema)
```

What each column kind needs in the schema is shown in
[examples/customers_schema.json](examples/customers_schema.json):

| Kind | Fields |
|---|---|
| `integer`, `float` | `minimum`/`maximum` for uniform values, or `mean`/`std`, kept inside the bounds when given; `unique` for identifiers |
| `boolean` | `true_share` |
| `category` | `choices`, optional `weights` |
| `datetime` | `start`, `end` |
| `text` | `category` for realistic fakes (`email`, `person_name`, `cpf`, `cnpj`, `phone`, ...), a `template` where `#` is a digit and `?` an uppercase letter, or `choices` |
| any | `null_fraction`, `unique` |

From a sample table (`reference`):

- **`statistical`**: a Gaussian copula learns each column's distribution and the
  correlations between numeric, date and category columns. Numbers are interpolated
  between observed quantiles, so rows are new rather than copied. Identifiers are always
  regenerated as fakes, never sampled from the reference.
- **`privacy`**: every column is drawn independently from an inferred schema, so no
  row-level structure survives. Categories seen fewer than 5 times become `OTHER`.

```python
synthetic = anonymizer.synthesize(5_000, reference=customers, fidelity=Fidelity.STATISTICAL)
```

## Development

```bash
uv sync
uv run pytest
uv run ruff format .
uv run ruff check .
uv run mypy src
```

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md): how the code is organised and where to change things
- [CHANGELOG.md](CHANGELOG.md): released changes
