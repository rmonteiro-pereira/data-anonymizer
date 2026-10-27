# Detection

`anonymize scan` and `Anonymizer.scan()` classify every column. Each finding has:
- a storage kind;
- a `DataCategory`;
- a `Sensitivity`;
- a confidence between 0 and 1;
- the evidence, in one sentence.

## Categories

| Sensitivity | Meaning | Categories |
|---|---|---|
| direct identifier | Points at one person on its own | email, phone, person_name, cpf, cnpj, ssn, credit_card, iban, ip_address, street_address, cnh, nis, health_card, voter_id, vehicle_registration, vehicle_plate, pix_key, invoice_key, national_id, imei, mac_address, record_id, free_text_with_pii |
| quasi-identifier | Identifies in combination | url, postal_code, city, date_of_birth, age, gender, geo_coordinate |
| sensitive | What an attacker wants to learn | income, health |
| non-sensitive | Nothing found | generic_numeric, generic_category, generic_datetime, generic_text |

A CNPJ is kept as a direct identifier, because for a sole proprietor (MEI) it identifies a
person under the LGPD.

## Recognized values

| Identifier | Rule checked |
|---|---|
| CPF | Two mod-11 check digits |
| CNPJ | Two mod-11 check digits. The alphanumeric format in use since July 2026 is supported: each letter counts as its ASCII code minus 48 |
| CNH | Two check digits with the offset rule |
| PIS / PASEP / NIS / NIT | Mod-11 check digit, weights 3,2,9,8,7,6,5,4,3,2 |
| RENAVAM | Mod-11 check digit. The check is equivalent to NIS, so it is only chosen when the column name says `renavam` |
| CNS (health card) | Mod-11 rule for cards starting 7–9; reconstruction rule for cards starting 1–2 |
| Título de Eleitor | State code 01–28 and two check digits. Both the general rule and the SP/MG remainder rule are accepted |
| NF-e access key | 44 digits, mod-11 check digit |
| Vehicle plate | Old and Mercosul formats; needs a name hint (`placa`, `plate`) |
| PIX random key | UUID v4; needs a name hint (`pix`, `chave_pix`), otherwise a unique UUID column is a record ID |
| CEP | `00000-000` |
| Phone | Layouts with fixed digit groups: North American 3-3-4 with any separators or brackets, `+<country>` numbers in any grouping (including compact E.164), Brazilian, Portuguese, British, Australian and French groupings. Formatted numbers need 10–13 digits. Bare digit strings only when they look like a Brazilian mobile (area code then 9), because bare 10–12 digit numbers are usually timestamps or order numbers. CPF, CNPJ, IPv4 addresses, version strings and dates never match |
| Payment card | Luhn, a known network prefix, and that network's length. 15-digit numbers must start 34 or 37, which keeps IMEIs out |
| IMEI | 15 digits with a Luhn check digit |
| IBAN | ISO 13616 mod-97, with or without spaces |
| US SSN / ITIN | Issued ranges; ITIN group 70–88 or 90–99 except 93 |
| IPv4 / IPv6 | Parsed by the standard library |
| MAC address | Six hexadecimal pairs |
| Aadhaar | 12 digits, Verhoeff check |
| Spanish DNI / NIE | Mod-23 check letter |
| Portuguese NIF | Mod-11 check digit |
| Dutch BSN | Eleven-test |
| German tax ID (Steuer-IdNr) | Digit repetition rule and ISO 7064 MOD 11,10 |
| UK National Insurance number | Allowed prefixes and an A–D suffix |
| Italian codice fiscale | Check letter |
| Email, URL | Format |

These have no value check and are found by column name only: person names, street
addresses, cities, RG and passport numbers, age, gender, coordinates, income and
health. The checks are in `src/data_anonymizer/detection/checksums.py`.

## How a column is classified

1. **Storage kind.** It comes from the dtype and values: integer, float, boolean,
   category, datetime or text. Whole-number floats count as integers, and text that
   parses as dates counts as dates.
2. **Name hints.** The column name is split into tokens, with accents removed and
   camelCase split. It is then matched against English and Portuguese hints, such as
   `dataNascimento` → date of birth, or `cel` → phone. Negative hints, such as `order`,
   `pedido`, `protocolo`, `sku`, `timestamp` and `nsu`, mark business codes. In those
   columns, bare digit strings are never treated as documents or phones.
3. **Value evidence.** Every recognizer runs on a sample of up to 1000 values. Integer
   columns get their lost leading zeros back first. A recognizer that needs a name hint
   only runs when the name points to it. A category passes when:
   - 80% of values pass its checksum, or
   - 90% match its format, when the category has no checksum.
4. **Confidence tiers.**
   - Values pass a checksum: the confidence is the share of passing values.
   - The name also agrees: plus 0.15.
   - A format match only: 0.8 times the share.
   - A runner-up within 0.05 of the winner, with no name to settle it: the evidence
     names both, and confidence drops by 0.25.
5. **Fallbacks, in order.**
   1. A name hint alone gives 0.7. When the values contradict a checkable hint, the
      confidence is 0.4 and the plan asks for review.
   2. Text averaging at least two words per value, where 10% of values embed personal
      data, is classed as free text with personal data. Recognizers that random numbers
      in prose would trigger, such as CNH, NIS or CNS, are not used inside text.
   3. Gender labels.
   4. Dates in the past, spread over decades and with no time of day, are birth dates
      (0.5).
   5. Nearly unique short values are a record ID (0.5).

Findings below 0.6 confidence get "please review" in the proposed plan.
