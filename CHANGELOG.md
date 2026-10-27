# Changelog

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
