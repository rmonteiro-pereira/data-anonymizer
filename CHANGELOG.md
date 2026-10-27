# Changelog

## Unreleased

- Pseudonymization and anonymization as explicit protection targets. Anonymization plans
  refuse linking techniques and run under a throwaway key. Every result gets a protection
  level (anonymized, at_risk, pseudonymized, identifiable). The level comes from record
  isolation, linkage and inference checks, and `apply` exits with code 3 when the target
  is missed.
- `surrogate` gains `consistent: false` for random fakes per cell.
- Detection:
  - New identifiers:
    - Brazil: alphanumeric CNPJ, CNH, PIS/PASEP/NIS, CNS, Título de Eleitor, RENAVAM,
      NF-e keys, vehicle plates and PIX keys;
    - international: IMEI, IPv6, MAC, Aadhaar, Spanish DNI/NIE, Portuguese NIF, Dutch
      BSN, German tax ID, UK NINO, US ITIN and Italian codice fiscale.
  - Scored decisions with tie flags, negative name hints, leading-zero recovery, and
    birth dates recognized from their values.
  - Fixes: bare 10–12 digit numbers (timestamps, order numbers) were classed as phones,
    and IMEIs as payment cards.
- Noise explorer: an interactive page per column, from `explore-noise`. Disclosure
  metrics: record linkage, interval disclosure, per-record shift, and Wasserstein / IQR.
- Synthesis patterns:
  - distributions (lognormal, gamma, exponential, Poisson, quantile curves);
  - Spearman correlations, conditional overrides and formula columns, with a safe
    expression language that includes date and random functions;
  - outliers, exact statistics and row constraints.
- Synthesis fixes:
  - the expression language rejects huge powers, text repetition and over-long or deeply
    nested expressions, and reports evaluation errors as `ExpressionError`;
  - a discrete distribution (Poisson) keeps its `minimum` value when truncated, and the
    end values of an integer column's bounds are as likely as the rest;
  - conditional overrides are validated after merging with the column: a mean and std
    replace a uniform or quantile base shape, and new choices reset the weights;
- Data-science challenges, from `challenge`: train and test files, plus an answer key
  with the true drivers and the best achievable scores.

## 0.1.0

First release.

- Column detection from names (English and Portuguese) and values, with checksum
  validation for CPF, CNPJ, card numbers and IBAN.
- Editable JSON anonymization plans, proposed under `statistical` or `privacy` fidelity.
- 26 techniques across masking, pseudonymization, encryption, generalization, noise and
  perturbation, including Argon2id and bcrypt hashing, a choice of HMAC algorithm, and
  reversible AES-SIV, AES-GCM and FF1 format-preserving encryption.
- `reveal` (Python and CLI) to decrypt reversible columns with the secret key.
- k-anonymity enforcement by local suppression; k-anonymity, l-diversity and drift reports.
- Synthetic data from a hand-written schema, or from a reference table through a Gaussian
  copula.
- The `anonymize` command line.
