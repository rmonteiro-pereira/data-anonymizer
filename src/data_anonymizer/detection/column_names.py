"""Guess what a column holds from its name, in English and Portuguese.

Names are a weaker signal than values: `name` may be a product name. The detector
therefore gives a name-only guess a lower confidence than a guess the values confirm.
"""

import re
import unicodedata

from data_anonymizer.taxonomy import DataCategory

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
    DataCategory.CNPJ: ("cnpj",),
    DataCategory.SSN: ("ssn", "social_security"),
    DataCategory.CREDIT_CARD: ("credit_card", "card_number", "cc_number", "cartao"),
    DataCategory.IBAN: ("iban",),
    DataCategory.IP_ADDRESS: ("ip", "ip_address", "ipv4", "ipv6"),
    DataCategory.URL: ("url", "website", "link"),
    DataCategory.POSTAL_CODE: ("zip", "zipcode", "zip_code", "postal_code", "postcode", "cep"),
    DataCategory.STREET_ADDRESS: ("address", "street", "endereco", "logradouro"),
    DataCategory.CITY: ("city", "cidade", "municipio", "town", "mun"),
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


def has_negative_hint(column_name: str) -> bool:
    """True when the name says the column holds business codes, counters or timestamps."""
    tokens = name_tokens(column_name)
    joined = "_".join(tokens)
    return any(token in _NEGATIVE_TOKENS for token in tokens) or joined in _NEGATIVE_TOKENS


def category_from_name(column_name: str) -> DataCategory | None:
    """Return the category a column name suggests, or None when it suggests nothing."""
    tokens = name_tokens(column_name)
    if not tokens:
        return None

    joined = "_".join(tokens)
    if joined in _EXACT_NAMES:
        return _EXACT_NAMES[joined]

    for category, hints in _HINTS.items():
        for hint in hints:
            if _contains_sequence(tokens, hint.split("_")):
                return category

    if tokens[-1] in _ID_TOKENS or tokens[0] in _ID_TOKENS:
        return DataCategory.RECORD_ID
    return None
