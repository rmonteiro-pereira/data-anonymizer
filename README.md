# data-anonymizer

A Python library and command-line tool for tables that contain personal data. It:

1. **detects** which columns hold personal or sensitive data. It reads column names and
   values, and checks the check digits of about 30 document formats, Brazilian and
   international;
2. **proposes** a plan, a JSON file you can review and edit, that assigns one of 28
   techniques to each flagged column;
3. **applies** the plan as **pseudonymization** (keyed, re-linkable by whoever holds the key)
   or **true anonymization** (no link left, and checked; the default), keeping either the
   statistics or the privacy first;
4. **shows how noise deflects your data** on an interactive page: distributions, per-record
   shifts and re-identification risk for every noise technique and strength;
5. **reveals** the original values of reversibly encrypted columns, for whoever holds the key;
6. **generates** synthetic data from a sample or from a schema. A schema can plant
   correlations, conditional rules, formulas, trends, seasonality, outliers and exact
   statistics. It can also generate **data-science challenges with a known answer key**.

## Features and algorithms

| Area | What is implemented |
|---|---|
| Detection | Patterns with checksum validation. **Brazil:** CPF, CNPJ (numeric and the new alphanumeric format), CNH, PIS/PASEP/NIS, CNS, Título de Eleitor, RENAVAM, NF-e key, vehicle plates, PIX keys, CEP, phones. **International:** cards (Luhn and issuer prefix), IBAN, US SSN and ITIN, IMEI, IPv4/IPv6, MAC, Aadhaar (Verhoeff), Spanish DNI/NIE, Portuguese NIF, Dutch BSN, German tax ID, UK NINO, Italian codice fiscale. Column-name hints in English and Portuguese, with negative hints for order numbers and timestamps. All candidates are scored, ties are flagged, and confidence comes in tiers. Also: leading zeros lost to integer storage are restored, birth dates are recognized from their values, and personal data embedded in free text is found |
| Masking, one-way | Redaction, suppression, partial masking that keeps separators, entity redaction inside free text |
| Hashing, one-way | HMAC with SHA-256, SHA-512, SHA3-256, SHA3-512, BLAKE2b or BLAKE2s. **Argon2id** and **bcrypt**, with a keyed salt (joins survive) or a random one |
| Pseudonymization | Keyed format-preserving scrambling. Realistic surrogates: names, emails, phones, valid CPF/CNPJ, SSN, cards, IBAN, IPs, addresses, URLs. A surrogate is either consistent per value (a pseudonym) or random per cell (no link) |
| Encryption, reversible | **AES-SIV** deterministic encryption (RFC 5297). **AES-256-GCM** randomized encryption. **FF1** format-preserving encryption (NIST SP 800-38G, checked against the NIST sample vectors). Keys derived from one secret with **HKDF-SHA256** |
| Generalization | Fixed-width, equal-width and quantile binning. Rounding. Date and string truncation. Top and bottom coding. Rare-category suppression |
| Noise | Gaussian noise rescaled to keep mean and variance. **Laplace mechanism** (ε-local differential privacy with fixed bounds). Multiplicative noise. Date shifting, with one offset per person or company |
| Perturbation | Shuffling, rank swapping, univariate microaggregation, **randomized response** (local differential privacy) |
| Business confidentiality | Opaque category codes in random order, with small segments merged. **Per-store indexing** by default, so no known value plus a public total gives the money back; recentring and rescaling (robust, z-score, min-max) when sizes must stay comparable; rank, log-ratio and growth too. Dominance and p% checks, so a segment's biggest contributor cannot be read from the release. See [docs/business-confidentiality.md](docs/business-confidentiality.md) |
| Protection modes | **Anonymization** (the default) or **pseudonymization**, each with `statistical` or `privacy` fidelity. Anonymization refuses linking techniques and runs under a throwaway key. Identifiers are checked on the data, not only by technique name, and unique columns released unchanged are flagged. Each result is labelled `anonymized`, `at_risk`, `pseudonymized` or `identifiable`, by checks named after the EDPB 2026 criteria: no record isolation, no linkage, no inference |
| Privacy and risk | k-anonymity, l-diversity, k-anonymity enforcement by local suppression, whole-record uniqueness, distance-based record linkage, interval disclosure |
| Utility | Change in mean and standard deviation, Kolmogorov–Smirnov, Wasserstein / IQR, total variation distance, correlation change, per-record shift in standard deviations |
| Noise explorer | A self-contained interactive HTML page per column. For every technique and strength it shows before/after distributions, original against released per record with an exposure band, and a risk-utility map. It works offline and follows light and dark themes |
| Synthesis | Distributions (normal, lognormal, gamma, exponential, Poisson, uniform, quantile curves) truncated to bounds. Spearman correlations imposed through a Gaussian copula. Conditional overrides that keep correlations. Formula columns using a safe expression language with date and random functions. Outlier planting, exact statistics, row constraints, unique keys, null shares and templates. Learned Gaussian copula from a reference table |
| Challenges | A schema `question` produces train and test files and a private answer key. The key lists the true drivers, the correlated-but-not-causal columns and the distractors, plus the best score any model can reach (RMSE/R², or accuracy/AUC/log loss). Columns that leak the target are flagged. With a `treatment`, the key also holds the true average, treated-only and per-group effects, the bias of the naive comparison, the confounders, mediators, colliders and instruments, and a valid adjustment set. Optional test-set drift (covariate or concept) and label noise on the train or test labels, both reported in the key. Panels of entities over periods (entity-level and period-level columns, `lag`, AR series, seasonality, regimes) with a time split that holds out the last periods |
| Interfaces | Python API (`Anonymizer`), editable JSON plans and schemas, the `anonymize` CLI, CSV and Parquet, custom techniques |

### Roadmap

Planned, not yet implemented:

- **Richer data-science challenges.**
  - Class imbalance and missing-not-at-random patterns.
  - Multi-table relational data with foreign keys.
  - Difficulty knobs and question text generated from the schema.
- **Detection.**
  - An optional named-entity extra for names and addresses inside text: GLiNER with
    `urchade/gliner_multi_pii-v1` or a Portuguese PII model, or Presidio.
  - Names and cities recognized from their values, using IBGE name and municipality lists.
  - An optional `python-stdnum` extra for more countries, and `phonenumbers` for
    international phones.
  - Passport MRZ, crypto wallet addresses, and confidence calibrated on a labelled benchmark.
- **Tokenization with a vault**: random tokens with the mapping kept in a separate store.
- **Multi-table consistency**: keep pseudonyms and date offsets consistent across several
  tables in one call.
- **Stronger anonymization**:
  - Mondrian multidimensional generalization;
  - generalization hierarchies (postal code → region → state);
  - multivariate microaggregation (MDAV) and t-closeness;
  - l-diversity enforcement;
  - attack-based checks with Anonymeter (singling out, linkability, inference).
- **Differential privacy**: an optional OpenDP extra for DP aggregates, and DP synthesis
  (MST, AIM) with an ε budget across columns.
- **Synthetic-data checks**: distance to the closest real record and membership inference.
- **Scale**: Spark tables through the same engine layer (see
  [docs/adr/0001-dataframe-backends.md](docs/adr/0001-dataframe-backends.md)), and chunked
  processing for files larger than memory.

## Quick start

```bash
git clone https://github.com/rmonteiro-pereira/data-anonymizer.git
cd data-anonymizer
uv sync
export ANONYMIZER_SECRET_KEY="$(openssl rand -hex 32)"

uv run anonymize scan customers.csv
uv run anonymize propose customers.csv --protection anonymization -o plan.json
# review and edit plan.json, then:
uv run anonymize apply customers.csv --plan plan.json -o customers_anon.csv --report report.json
uv run anonymize explore-noise customers.csv --column salary -o salary_noise.html
```

The same flow in Python. `scan`, `propose`, `validate`, `apply` and `reveal` take pandas,
Polars and DuckDB tables and return the same kind. A DuckDB table needs the plan's
`row_key` (see [docs/protection.md](docs/protection.md)).

```python
import pandas as pd
from data_anonymizer import Anonymizer, Fidelity, Protection

customers = pd.read_csv("customers.csv")
anonymizer = Anonymizer(secret_key="a long secret", seed=42, locale="pt_BR")

for finding in anonymizer.scan(customers):
    print(finding.column, finding.category, finding.confidence, finding.evidence)

plan = anonymizer.propose(
    customers, protection=Protection.ANONYMIZATION, fidelity=Fidelity.STATISTICAL
)
result = anonymizer.apply(customers, plan)
result.data  # the anonymized DataFrame, of the kind you passed in
result.protection.level  # anonymized, at_risk, pseudonymized or identifiable
result.protection.problems  # why a level was not reached
result.utility.max_correlation_change
```

## Configuration

| Variable | Required | Default | What it does |
|---|---|---|---|
| `ANONYMIZER_SECRET_KEY` | for `reveal` | random per run | At least 16 bytes; a shorter key is refused. Keys every pseudonym and keyed salt, and is the master secret for the encryption keys. With the same key, the same value always gets the same pseudonym, so tables pseudonymized separately still join. `reveal` refuses to run without it. `apply` warns and uses a random key, so pseudonyms change on every run and encrypted columns can never be revealed. Anonymization plans use a throwaway key instead for everything that could link a release back or to another release: pseudonyms, encryption, category codes and random fakes. The key still drives their noise, which is reproducible for the key holder, so keep it secret even then. |

In Python, pass `secret_key=` to `Anonymizer`; the library never reads the environment.
Treat the key like a password. Anyone who holds it can decrypt the reversible columns,
and can test whether a guessed real value produces a given pseudonym.

## Command line

All commands take `--seed N` (reproducible noise and synthesis; with a secret key the
noise is reproducible only for the key's holder, with a seed alone anyone who knows the
seed can recompute it) and `--locale` (Faker
locale for fake values, such as `pt_BR`; default `en_US`). Tables can be `.csv` or
`.parquet`, read and written with Polars.

| Command | What it does |
|---|---|
| `anonymize scan INPUT [--sample-size N]` | Prints each column's category, sensitivity, confidence and evidence |
| `anonymize propose INPUT [--protection P] [--fidelity F] [-o PLAN]` | Writes a plan as JSON (stdout without `-o`) |
| `anonymize apply INPUT -o OUTPUT [--plan PLAN] [--protection P] [--fidelity F] [--report REPORT] [--row-key COLUMN]` | Applies a plan, or proposes one first. Prints the protection level reached. `--report` writes privacy, utility and protection results as JSON. `--row-key` overrides the plan's `row_key` |
| `anonymize reveal INPUT --plan PLAN -o OUTPUT [--row-key COLUMN]` | Decrypts the columns the plan encrypted and leaves one-way columns as they are |
| `anonymize explore-noise INPUT --column NAME -o PAGE.html [--band SD] [--max-rows N]` | Writes the interactive noise explorer for one numeric column. Columns longer than `N` values (default 200,000) are explored on a random sample |
| `anonymize infer-schema INPUT [--fidelity F] [-o SCHEMA]` | Learns a generation schema from a table |
| `anonymize synthesize -o OUTPUT [--rows N] [--schema SCHEMA] [--reference INPUT] [--fidelity F]` | Generates a synthetic table |
| `anonymize challenge --schema SCHEMA [--rows N] -o FOLDER` | Generates a data-science challenge: `train.csv`, `test.csv`, `QUESTION.md`, and `answers/` with the answer key |
| `anonymize techniques` | Lists every technique, its parameters and whether it is reversible |

`P` is `anonymization` (the default) or `pseudonymization`. `F` is `statistical` (the
default) or `privacy`. `-o` is short for `--output`.

Exit codes:
- `0`: success.
- `2`: an error.
- `3`: `apply` wrote its output, but the result does not reach the plan's protection target.

## Documentation

- [docs/detection.md](docs/detection.md): categories, recognized identifiers, and how a
  column is classified
- [docs/protection.md](docs/protection.md): pseudonymization against anonymization, the
  default proposals, plan files and the protection checks
- [docs/techniques.md](docs/techniques.md): every technique, its parameters and its effect
- [docs/business-confidentiality.md](docs/business-confidentiality.md): sharing company data,
  such as tenant revenue and rent, without revealing any one company's figures
- [docs/noise-explorer.md](docs/noise-explorer.md): reading the noise explorer and its metrics
- [docs/synthesis.md](docs/synthesis.md): schemas, planted patterns, synthesis from a
  sample, and data-science challenges
- [ARCHITECTURE.md](ARCHITECTURE.md): how the code is organised and where to change things
- [CHANGELOG.md](CHANGELOG.md): released changes

## Development

```bash
uv sync
uv run pytest
uv run ruff format .
uv run ruff check .
uv run mypy src
```
