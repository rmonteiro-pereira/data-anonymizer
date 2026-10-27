"""Guess what a column holds from its name, in English and Portuguese.

Names are a weaker signal than values: `name` may be a product name. The detector
therefore gives a name-only guess a lower confidence than a guess the values confirm.
A name that also contains a qualifier, such as `email_count` or `address_type`, says what
kind of thing the column holds about the category, so it suggests no category at all.
"""

import re
import unicodedata

from oblitus.taxonomy import DataCategory

# Each hint is a sequence of name tokens that must appear next to each other.
# "birth" matches `birth_date` and `dateOfBirth`; "ip" does not match `zip`.
_HINTS: dict[DataCategory, tuple[str, ...]] = {
    DataCategory.EMAIL: ("email", "e_mail", "mail"),
    DataCategory.PHONE: (
        "phone",
        "telephone",
        "mobile",
        "cell",
        "celular",
        "telefone",
        "fone",
        "tel",
        "cel",
        "whatsapp",
    ),
    DataCategory.PERSON_NAME: (
        "first_name",
        "last_name",
        "full_name",
        "firstname",
        "lastname",
        "surname",
        "nome_completo",
        "sobrenome",
        "customer_name",
        "client_name",
        "patient_name",
        "employee_name",
        "contact_name",
        "nome_cliente",
    ),
    DataCategory.ORGANIZATION: (
        "razao_social",
        "nome_fantasia",
        "company_name",
        "empresa",
        "tenant",
        "lojista",
        "store_name",
        "nome_loja",
        "brand",
        "marca",
        "merchant",
    ),
    DataCategory.CPF: ("cpf",),
    DataCategory.CNH: ("cnh", "carteira_motorista", "drivers_license", "driver_license"),
    DataCategory.NIS: ("pis", "pasep", "nis", "nit"),
    DataCategory.HEALTH_CARD: ("cns", "cartao_sus", "sus"),
    DataCategory.VOTER_ID: ("titulo_eleitor", "titulo_eleitoral", "voter_id", "voter"),
    DataCategory.VEHICLE_REGISTRATION: ("renavam",),
    DataCategory.VEHICLE_PLATE: ("placa", "plate", "license_plate", "licence_plate"),
    DataCategory.PIX_KEY: ("pix", "chave_pix"),
    DataCategory.INVOICE_KEY: ("nfe", "nf_e", "chave_nfe", "chave_acesso", "access_key"),
    DataCategory.IMEI: ("imei",),
    DataCategory.MAC_ADDRESS: ("mac", "mac_address"),
    DataCategory.NATIONAL_ID: (
        "rg",
        "identidade",
        "national_id",
        "dni",
        "nie",
        "nif",
        "nino",
        "bsn",
        "aadhaar",
        "itin",
        "steuer_id",
        "codice_fiscale",
        "passport",
        "passaporte",
    ),
    DataCategory.CRYPTO_WALLET: (
        "wallet",
        "carteira_cripto",
        "bitcoin",
        "btc",
        "ethereum",
        "eth_address",
        "crypto_address",
    ),
    DataCategory.CNPJ: ("cnpj",),
    DataCategory.SSN: ("ssn", "social_security"),
    DataCategory.CREDIT_CARD: ("credit_card", "card_number", "cc_number", "cartao"),
    DataCategory.IBAN: ("iban",),
    DataCategory.IP_ADDRESS: ("ip", "ip_address", "ipv4", "ipv6"),
    DataCategory.URL: ("url", "website", "link"),
    DataCategory.POSTAL_CODE: ("zip", "zipcode", "zip_code", "postal_code", "postcode", "cep"),
    DataCategory.STREET_ADDRESS: (
        "address",
        "street",
        "endereco",
        "logradouro",
        "rua",
        "addr",
        "address1",
        "address2",
        "morada",
    ),
    DataCategory.CITY: ("city", "cidade", "municipio", "town", "mun", "bairro"),
    # Nationality comes before date of birth, so `country_of_birth` is not read as a birth date.
    DataCategory.NATIONALITY: (
        "nationality",
        "nacionalidade",
        "native_country",
        "country_of_birth",
        "naturalidade",
    ),
    DataCategory.SPECIAL_CATEGORY: (
        "race",
        "raca",
        "cor_raca",
        "etnia",
        "ethnicity",
        "religion",
        "religiao",
        "orientacao_sexual",
        "sexual_orientation",
        "political",
        "partido",
        "sindicato",
        "union_member",
        "biometric",
        "genetic",
    ),
    DataCategory.MARITAL_STATUS: ("marital_status", "estado_civil"),
    DataCategory.DATE_OF_BIRTH: (
        "dob",
        "birth",
        "birthdate",
        "birthday",
        "nascimento",
        "nasc",
    ),
    DataCategory.AGE: ("age", "idade"),
    DataCategory.GENDER: ("gender", "sex", "sexo", "genero"),
    DataCategory.GEO_COORDINATE: ("lat", "latitude", "lon", "lng", "longitude"),
    DataCategory.INCOME: ("salary", "income", "wage", "salario", "renda", "remuneracao"),
    DataCategory.HEALTH: ("diagnosis", "diagnostico", "disease", "doenca", "cid10"),
    # Revenue and rent come before floor area, so `sales_per_sqm` is revenue.
    DataCategory.BUSINESS_REVENUE: (
        "revenue",
        "sales",
        "faturamento",
        "vendas",
        "receita",
        "gmv",
        "turnover",
    ),
    DataCategory.RENT: ("rent", "aluguel", "condominio", "occupancy_cost", "custo_ocupacao"),
    DataCategory.FLOOR_AREA: ("abl", "gla", "sqm", "m2", "metragem", "area_m2", "floor_area"),
    DataCategory.BUSINESS_SEGMENT: (
        "segmento",
        "segment",
        "ramo",
        "setor",
        "store_category",
        "categoria_loja",
    ),
}

# Words that name a person's role. Combined with a name word they make a person-name column.
_NAME_WORDS = {"nome", "name", "nm"}
_PERSON_WORDS = {
    "cliente",
    "customer",
    "client",
    "paciente",
    "patient",
    "mae",
    "pai",
    "mother",
    "father",
    "responsavel",
    "titular",
    "funcionario",
    "employee",
    "owner",
    "holder",
    "buyer",
    "seller",
    "sender",
    "recipient",
    "student",
    "doctor",
    "author",
    "given",
    "family",
    "first",
    "last",
    "social",
}
_STOP_WORDS = {"do", "da", "de", "dos", "das", "of", "the", "o", "a"}

# Words that turn a column into a count, a label or a measure about the category they follow.
_QUALIFIER_TOKENS = {
    "count",
    "qtd",
    "quantidade",
    "type",
    "tipo",
    "status",
    "flag",
    "score",
    "brand",
    "ratio",
    "verified",
    "calls",
    "total",
    "rate",
}

# Descriptive attributes: an `_id` column next to them holds keys, not the attribute itself.
_YIELD_TO_ID_TOKENS = {
    DataCategory.ORGANIZATION,
    DataCategory.NATIONALITY,
    DataCategory.SPECIAL_CATEGORY,
    DataCategory.MARITAL_STATUS,
    DataCategory.BUSINESS_REVENUE,
    DataCategory.RENT,
    DataCategory.FLOOR_AREA,
    DataCategory.BUSINESS_SEGMENT,
}

# A bare `name` or `nome` column is nearly always a person; `product_name` is not.
_EXACT_NAMES: dict[str, DataCategory] = {
    "name": DataCategory.PERSON_NAME,
    "nome": DataCategory.PERSON_NAME,
}

_ID_TOKENS = {"id", "uuid", "guid"}

# Names of columns that hold business codes, counters or timestamps. In these columns only
# recognizers with a strong check run, so a part number is not taken for a phone number.
_NEGATIVE_TOKENS = {
    "order",
    "pedido",
    "protocolo",
    "protocol",
    "nsu",
    "sku",
    "ts",
    "timestamp",
    "epoch",
    "produto",
    "product",
    "item",
    "lote",
    "batch",
    "version",
    "versao",
    "serial",
    "invoice_number",
    "ref",
    "reference",
    "referencia",
    "code",
    "codigo",
    "part",
    "claim",
}


def name_tokens(column_name: str) -> list[str]:
    """Split a column name into lowercase ASCII tokens on separators and camelCase."""
    without_accents = (
        unicodedata.normalize("NFKD", column_name).encode("ascii", "ignore").decode("ascii")
    )
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", without_accents)
    return [token for token in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if token]


def _contains_sequence(tokens: list[str], hint_tokens: list[str]) -> bool:
    width = len(hint_tokens)
    return any(tokens[start : start + width] == hint_tokens for start in range(len(tokens)))


def _significant_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in _STOP_WORDS]


def has_negative_hint(column_name: str) -> bool:
    """True when the name says the column holds business codes, counters or timestamps."""
    tokens = name_tokens(column_name)
    joined = "_".join(tokens)
    return any(token in _NEGATIVE_TOKENS for token in tokens) or joined in _NEGATIVE_TOKENS


def is_person_word_column(column_name: str) -> bool:
    """True when the name is only person words, such as `customer` or `titular`.

    Such a column holds person names or something else about people; only its values
    can tell which.
    """
    tokens = _significant_tokens(name_tokens(column_name))
    return bool(tokens) and all(token in _PERSON_WORDS for token in tokens)


def category_from_name(column_name: str) -> DataCategory | None:
    """Return the category a column name suggests, or None when it suggests nothing."""
    tokens = name_tokens(column_name)
    if not tokens:
        return None

    joined = "_".join(tokens)
    if joined in _EXACT_NAMES:
        return _EXACT_NAMES[joined]

    hinted = _hinted_category(tokens)
    if hinted is not None:
        category, matched_tokens = hinted
        if _has_unmatched_qualifier(tokens, matched_tokens):
            return None
        if category in _YIELD_TO_ID_TOKENS and _has_id_token(tokens):
            return DataCategory.RECORD_ID
        return category

    if _has_id_token(tokens):
        return DataCategory.RECORD_ID
    return None


def _hinted_category(tokens: list[str]) -> tuple[DataCategory, list[str]] | None:
    if _is_person_name(tokens):
        return DataCategory.PERSON_NAME, [token for token in tokens if token in _NAME_WORDS]
    for category, hints in _HINTS.items():
        for hint in hints:
            hint_tokens = hint.split("_")
            if _contains_sequence(tokens, hint_tokens):
                return category, hint_tokens
    return None


def _is_person_name(tokens: list[str]) -> bool:
    significant = _significant_tokens(tokens)
    has_name_word = any(token in _NAME_WORDS for token in significant)
    has_person_word = any(token in _PERSON_WORDS for token in significant)
    return has_name_word and has_person_word


def _has_unmatched_qualifier(tokens: list[str], matched_tokens: list[str]) -> bool:
    return any(token in _QUALIFIER_TOKENS and token not in matched_tokens for token in tokens)


def _has_id_token(tokens: list[str]) -> bool:
    return tokens[-1] in _ID_TOKENS or tokens[0] in _ID_TOKENS
