# Detection

`oblitus scan` and `Anonymizer.scan()` classify every column. Each finding has:
- a storage kind;
- a `DataCategory`;
- a `Sensitivity`;
- a confidence between 0 and 1, calibrated on a labelled benchmark (see below);
- the evidence, in one sentence.

## Categories

| Sensitivity | Meaning | Categories |
|---|---|---|
| direct identifier | Points at one person, or one company, on its own | email, phone, person_name, cpf, cnpj, ssn, credit_card, iban, ip_address, street_address, cnh, nis, health_card, voter_id, vehicle_registration, vehicle_plate, pix_key, invoice_key, national_id, imei, mac_address, crypto_wallet, organization, record_id, free_text_with_pii |
| quasi-identifier | Identifies in combination | url, postal_code, city, date_of_birth, age, gender, geo_coordinate, nationality, marital_status, business_segment, floor_area |
| sensitive | What an attacker wants to learn | income, health, special_category (race, religion, politics, union, sexual orientation, biometric, genetic), business_revenue, rent |
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
| RENAVAM | Mod-11 check digit. The check is equivalent to NIS, so it is only chosen when the column name says `renavam`. The legacy 9-digit form is padded with two zeros first |
| CNS (health card) | Mod-11 rule for cards starting 7–9; reconstruction rule for cards starting 1–2 |
| Título de Eleitor | State code 01–28 and two check digits. Both the general rule and the SP/MG remainder rule are accepted |
| NF-e access key | 44 digits, mod-11 check digit |
| Vehicle plate | Old and Mercosul formats; needs a name hint (`placa`, `plate`) |
| PIX random key | UUID v4; needs a name hint (`pix`, `chave_pix`), otherwise a unique UUID column is a record ID |
| CEP | `00000-000` |
| Phone | Layouts with fixed digit groups: North American 3-3-4 with any separators or brackets, `+<country>` numbers in any grouping (including compact E.164), Brazilian, Portuguese, British, Australian and French groupings. Formatted numbers need 10–13 digits. Bare digit strings only when they look like a Brazilian mobile (area code then 9), because bare 10–12 digit numbers are usually timestamps or order numbers. CPF, CNPJ, IPv4 addresses, version strings and dates never match |
| Payment card | Luhn, a known network prefix, and that network's length: Visa 13, 16 or 19; Mastercard (51–55, 2221–2720) 16; American Express (34, 37) 15; Diners Club 14–19; Discover, JCB, UnionPay, Elo, Hipercard and Maestro 16–19. 15-digit numbers must start 34 or 37, which keeps IMEIs out |
| IMEI | 15 digits with a Luhn check digit. Not searched for inside free text, where 15-digit order numbers pass the check one time in ten |
| IBAN | ISO 13616 mod-97, with or without spaces |
| US SSN / ITIN | Issued ranges; ITIN group 70–88 or 90–99 except 93 |
| IPv4 / IPv6 | Parsed by the standard library. Inside free text an IPv6 address needs at least three groups, so `::` and `std::vector` do not count |
| MAC address | Six hexadecimal pairs |
| Aadhaar | 12 digits, Verhoeff check |
| Spanish DNI / NIE | Mod-23 check letter |
| Portuguese NIF | Mod-11 check digit |
| Dutch BSN | Eleven-test |
| German tax ID (Steuer-IdNr) | Digit repetition rule and ISO 7064 MOD 11,10 |
| UK National Insurance number | Allowed prefixes and an A–D suffix |
| Italian codice fiscale | Check letter |
| Passport MRZ | Two lines of 44 characters (TD3), with the ICAO 9303 check digits of the passport number, birth date, expiry date, personal number and the whole line. The lines may be joined or split by a line break, and may be in lowercase |
| Bitcoin address | Base58check (starts 1 or 3, double SHA-256 checksum) or bech32 and bech32m (`bc1...`, polymod checksum; one case only) |
| Ethereum address | `0x` and 40 hex digits. Mixed-case addresses must pass the EIP-55 checksum; all-lowercase or all-uppercase ones carry none and pass on shape |
| City | The value is one of the 5,570 IBGE municipality names (about 5,300 distinct), ignoring case and accents. Not searched for in free text as a bare word. Names of two or more words ("Rio de Janeiro"), and one-word names after "em", "para" or "cidade de", are found in prose (lowercase multi-word names too, after "em", "para", "na", "no" or "cidade de") and redacted as `<CITY>`; one-word names that are also everyday nouns ("Saúde") are ignored |
| Email, URL | Format |

IBAN, DNI/NIE, UK National Insurance, codice fiscale, alphanumeric CNPJ and vehicle plates are matched in upper and lower case.

These have no value check and are found by column name only: street addresses, cities
outside Brazil, RG and passport numbers, age, gender, coordinates, income, health, special categories,
nationality, marital status and the business figures (segment, revenue, rent, floor
area). Organizations are found by name only. Person names are found by name, and a
column named after a person (`customer`, `titular`, `owner`) also needs values shaped
like names. The checks are in `src/oblitus/detection/checksums.py`.

## How a column is classified

1. **Storage kind.** It comes from the dtype and values: integer, float, boolean,
   category, datetime or text. Whole-number floats count as integers, and text that
   parses as dates counts as dates. Placeholder text counts as missing first: blanks,
   `-`, `--`, `N/D`, `NA`, `n/a`, `null`, `none`, `nan`, `?`, `sem informação`, and
   numbers made of one repeated digit, such as `000.000.000-00`.
2. **Name hints.** The column name is split into tokens, with accents removed and
   camelCase split. It is then matched against English and Portuguese hints, such as
   `dataNascimento` → date of birth, or `cel` → phone. A name word (`nome`, `name`, `nm`)
   next to a person word (`cliente`, `paciente`, `mae`, `titular`, `owner`, `given`, ...),
   in any order and with `do`/`da`/`de`/`of` between, is a person name. A qualifier word
   (`count`, `qtd`, `type`, `tipo`, `status`, `flag`, `score`, `brand`, `ratio`,
   `verified`, `calls`, `total`, `rate`) beside a hint means the column describes the
   category rather than holding it, so `email_count` and `address_type` suggest nothing.
   Negative hints, such as `order`,
   `pedido`, `protocolo`, `sku`, `ref`, `code`, `timestamp` and `nsu`, mark business
   codes. In those columns only the recognizers with a strong check run (CPF, CNPJ, CNH,
   Título de Eleitor, CNS, IBAN, email, URL), so a part number is not taken for a phone,
   a version string for an IP address or a Luhn-valid reference for a card. A hint
   the name itself spells out stays active: `order_cpf` is still checked for CPFs.
3. **Value evidence.** Every recognizer runs on a sample of up to 1000 values. Digit-only
   values get their lost leading zeros back first, whether stored as integers or as text. A recognizer that needs a name hint
   only runs when the name points to it. A category passes when:
   - 80% of values pass its checksum, or
   - 90% match its format, when the category has no checksum, or
   - 50% of values pass a strong check (CPF, CNPJ, CNH, Título de Eleitor, CNS, IBAN,
     email, URL), with the confidence capped at 0.55 so the plan asks for review.

   A column that mixes strong direct identifiers, such as CPF and CNPJ, passes on the
   share of values that fit any of them. It takes the larger category, and the evidence
   names the split.
4. **Confidence tiers.**
   - Values pass a checksum: the confidence is the share of passing values.
   - The name also agrees: plus 0.15. When two candidates tie, the one the name agrees
     with wins.
   - A format match only: 0.8 times the share.
   - A runner-up within 0.05 of the winner, with no name to settle it: the evidence
     names both, and confidence drops by 0.25.
5. **Fallbacks, in order.**
   1. A column named after a person whose values are two to five alphabetic words in
      title case, such as `Ana da Silva`, is a person name (0.7).
   2. A name hint alone gives 0.55, so the plan always asks for review. When the values
      contradict a checkable hint, the confidence is 0.4. When the storage kind cannot
      hold the category, such as a boolean `email` or a float `sexo`, it is 0.3.
   3. Dates in the past that are mostly adult, spread over decades (a standard deviation
      of 8 to 30 years), with no time of day and a typical age under 100, are birth dates
      (0.5).
   4. Text averaging at least two words per value, where 10% of values embed personal
      data, is classed as free text with personal data. Recognizers that random numbers
      in prose would trigger, such as CNH, NIS, CNS or IMEI, are not used inside text. A
      recognizer that can fire by chance, such as phone or card, counts only when its hits
      are more than twice its chance rate.
   5. Gender labels.
   6. A strong check that matched more than 20% of values, but too few to pass: that
      category at 0.4, never "no sign of personal data".
   7. Nearly unique short values are a record ID (0.5). Integers count only when they look
      like keys: non-negative, and either in a range under ten times the row count, or
      codes of at least 6 digits of which 95% have the same length (account, phone or
      record numbers). Epoch timestamps, amounts in cents and populations stay numeric.

`scan` accepts object columns that hold lists, dicts, sets or arrays by reading their
text. It raises `TableFormatError` for duplicate column names and `ValueError` for a
`sample_size` below 1.

Findings below 0.6 confidence get "please review" in the proposed plan.

## Names and cities from their values

- **Cities.** A column whose values are mostly IBGE municipality names is a city, even when its
  name says nothing. A column named `city` whose values are not Brazilian is still a city, from
  the name alone.
- **Person names.** A column with no name hint whose values are two to five words in title case
  and start with a common Brazilian first name or end with a common surname is a person name (0.65 raw confidence), instead
  of a record ID. So is a column where at least 70% of the values hold a listed person name with
  at most one other word, such as `Paciente: Ana Silva`: with two or more other words the value
  is prose and the column is free text instead. The lists are small package data files in `src/oblitus/detection/data/`
  (see the README there): first names from the IBGE 2010 Census names ranking, municipalities
  from the IBGE localities API, and surnames from the Faker `pt_BR` provider, because IBGE
  publishes no surname ranking. They are public data. `benchmarks/build_reference_lists.py`
  rebuilds them.

## Optional extras

- **`identifiers`** (`pip install "oblitus[identifiers]"`). With `python-stdnum` installed, the
  detector also recognizes the Mexican CURP, Chilean RUT, Argentine CUIT/CUIL, Swedish
  personnummer, Danish CPR, Finnish HETU, Irish PPS number, Belgian national number, Swiss AHV
  number, French NIR and Indian PAN, each checked by `stdnum`. Only formats with punctuation,
  letters or a check digit are listed, so bare digit strings are not claimed. With
  `phonenumbers` installed, a `+` number that the library says is valid for its country counts
  as a phone even when the digit-count rule would reject it. Without the extras the built-in
  rules decide alone, and nothing is imported.
- **Names in prose, without any extra.** A capitalized first name from the bundled IBGE list
  followed by one or more capitalized words ("Ana Silva", "Maria da Silva") is a person name
  in text, and a municipality in prose is a city. A free-text column where at least 10% of the
  texts hold one of these among other words is classed as free text with personal data, so the
  default plan redacts it and the release check flags leftovers. A cell that holds only a name
  or only an address is still read as a name or an address column. Any run of spaces or tabs
  may separate the words of a name. A name that is not fully capitalized is found only when a
  listed first name is followed by a listed surname, in any mix of capitals ("ana silva",
  "ana Silva", "Ana silva", "maria da Silva"), and a lowercase municipality only when it has
  two or more words and follows "em", "para", "na", "no" or "cidade de" ("mora em rio de
  janeiro"); both lists are needed to keep ordinary lowercase prose out. In 20,000 generated
  sentences (5,000 each of lowercase Portuguese, lowercase English, mixed-case Portuguese and
  mixed-case English, built from ordinary nouns and a pool where 35% of the words are listed
  names), 211 lowercase Portuguese and 359 lowercase English sentences were flagged, and 2,050
  mixed-case Portuguese and 2,066 mixed-case English ones. Almost all hits are pairs of a listed
  first name and a listed word that is a name or surname, as in "lucia santos", which the rule
  is meant to find; the rate in real prose is far lower because few words there are names.
  The lists cover Brazilian names and places only. A foreign name ("Saoirse Murphy"), a
  Brazilian name outside the lists, a lowercase name with an unlisted surname and a
  one-word lowercase city are still missed; use the `ner` extra for those.
- **`ner`** (`pip install "oblitus[ner]"`). Free text can also be read by a named-entity model
  that finds names and addresses, which no pattern can. Set `entity_model` on a
  `redact_entities` action to `"gliner"` (`urchade/gliner_multi_pii-v1`) or `"presidio"`:
  `{"column": "note", "technique": "redact_entities", "params": {"entity_model": "gliner"}}`.
  Names become `<PERSON_NAME>` and addresses `<STREET_ADDRESS>`. Patterns keep their characters
  when a name overlaps them, and `categories` limits what the model may label. The model is
  downloaded and loaded when the first text is read, never at import, and the protection check
  on the released text asks the same model. A model can miss a name, so the report still lists
  the column as one where free-text details may remain. Any object with a
  `find(text) -> [(start, end, category)]` method can stand in for a model in code that calls
  `find_entities` directly. The extra pulls in PyTorch, so CI does not install it.

## Confidence calibration

The rule-based score (tiers above) is mapped to a probability with a monotone curve fitted by
isotonic regression on a labelled benchmark of 120 synthetic columns
(`benchmarks/detection_benchmark.py`): clean identifiers, identifiers with and without a name
hint, identifiers mixed with junk, names that promise a category the values do not keep, and
columns with nothing sensitive, all generated from the library's own surrogates. A reported
confidence of 0.9 therefore means that about nine in ten findings with that score were right
on the benchmark. The knots are constants in `detection/calibration.py`. After changing a
detection rule, refit them with `uv run python -m benchmarks.calibrate_detection`. A test
keeps the expected calibration error under 0.05 on the training columns and under 0.12 on a
second benchmark from another seed. `scan(..., calibrated=False)` returns the raw score. The
benchmark is synthetic and Brazil-leaning, so treat the numbers as a guide to how the tiers
rank, not as a measured error rate on your own data.
