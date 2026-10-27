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
  - Phone numbers are recognized in North American 3-3-4, compact E.164 and European
    layouts; CPF, CNPJ, IPv4 addresses, versions and dates no longer match.
  - Payment cards must have a length their network issues. IBAN, DNI/NIE, NINO, codice
    fiscale, alphanumeric CNPJ and plates match in lower case.
  - Free text: `::` and `std::vector` no longer count as IPv6, IMEI is not searched in
    prose, a recognizer must beat its chance rate, and the email pattern runs in linear time.
- Noise explorer: an interactive page per column, from `explore-noise`. Disclosure
  metrics: record linkage, interval disclosure, per-record shift, and Wasserstein / IQR.
- Synthesis patterns:
  - distributions (lognormal, gamma, exponential, Poisson, quantile curves);
  - Spearman correlations, conditional overrides and formula columns, with a safe
    expression language that includes date and random functions;
  - outliers, exact statistics and row constraints.
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
