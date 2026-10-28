"""The vocabulary shared by detection, proposal, techniques and synthesis."""

from enum import StrEnum


class ColumnKind(StrEnum):
    """The storage shape of a column, independent of what the values mean."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    DATETIME = "datetime"
    TEXT = "text"


NUMERIC_KINDS = frozenset({ColumnKind.INTEGER, ColumnKind.FLOAT})


class DataCategory(StrEnum):
    """What the values in a column mean, as far as privacy is concerned."""

    EMAIL = "email"
    PHONE = "phone"
    PERSON_NAME = "person_name"
    CPF = "cpf"
    CNPJ = "cnpj"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    URL = "url"
    STREET_ADDRESS = "street_address"
    CNH = "cnh"
    NIS = "nis"
    HEALTH_CARD = "health_card"
    VOTER_ID = "voter_id"
    VEHICLE_REGISTRATION = "vehicle_registration"
    VEHICLE_PLATE = "vehicle_plate"
    PIX_KEY = "pix_key"
    INVOICE_KEY = "invoice_key"
    NATIONAL_ID = "national_id"
    IMEI = "imei"
    MAC_ADDRESS = "mac_address"
    CRYPTO_WALLET = "crypto_wallet"
    ORGANIZATION = "organization"
    RECORD_ID = "record_id"
    FREE_TEXT_WITH_PII = "free_text_with_pii"
    POSTAL_CODE = "postal_code"
    CITY = "city"
    DATE_OF_BIRTH = "date_of_birth"
    AGE = "age"
    GENDER = "gender"
    GEO_COORDINATE = "geo_coordinate"
    NATIONALITY = "nationality"
    MARITAL_STATUS = "marital_status"
    BUSINESS_SEGMENT = "business_segment"
    FLOOR_AREA = "floor_area"
    INCOME = "income"
    HEALTH = "health"
    SPECIAL_CATEGORY = "special_category"
    BUSINESS_REVENUE = "business_revenue"
    RENT = "rent"
    GENERIC_NUMERIC = "generic_numeric"
    GENERIC_CATEGORY = "generic_category"
    GENERIC_DATETIME = "generic_datetime"
    GENERIC_TEXT = "generic_text"


class Sensitivity(StrEnum):
    """How much a column contributes to re-identifying or harming a person.

    Direct identifiers point at one person, or one company, on their own. Quasi-identifiers do so
    in combination (the classic example is birth date, postal code and gender).
    Sensitive attributes are what an attacker wants to learn about a person.
    """

    DIRECT_IDENTIFIER = "direct_identifier"
    QUASI_IDENTIFIER = "quasi_identifier"
    SENSITIVE = "sensitive"
    NON_SENSITIVE = "non_sensitive"


_DIRECT_IDENTIFIERS = {
    DataCategory.EMAIL,
    DataCategory.PHONE,
    DataCategory.PERSON_NAME,
    DataCategory.CPF,
    DataCategory.CNPJ,
    DataCategory.SSN,
    DataCategory.CREDIT_CARD,
    DataCategory.IBAN,
    DataCategory.IP_ADDRESS,
    DataCategory.STREET_ADDRESS,
    DataCategory.CNH,
    DataCategory.NIS,
    DataCategory.HEALTH_CARD,
    DataCategory.VOTER_ID,
    DataCategory.VEHICLE_REGISTRATION,
    DataCategory.VEHICLE_PLATE,
    DataCategory.PIX_KEY,
    DataCategory.INVOICE_KEY,
    DataCategory.NATIONAL_ID,
    DataCategory.IMEI,
    DataCategory.MAC_ADDRESS,
    DataCategory.CRYPTO_WALLET,
    DataCategory.ORGANIZATION,
    DataCategory.RECORD_ID,
    DataCategory.FREE_TEXT_WITH_PII,
}

_QUASI_IDENTIFIERS = {
    DataCategory.URL,
    DataCategory.POSTAL_CODE,
    DataCategory.CITY,
    DataCategory.DATE_OF_BIRTH,
    # Event dates (a visit, a purchase, an admission) single people out together with a
    # city or a sex, so they count like a birth date does.
    DataCategory.GENERIC_DATETIME,
    DataCategory.AGE,
    DataCategory.GENDER,
    DataCategory.GEO_COORDINATE,
    DataCategory.NATIONALITY,
    DataCategory.MARITAL_STATUS,
    DataCategory.BUSINESS_SEGMENT,
    DataCategory.FLOOR_AREA,
}

# Special categories are the data GDPR Article 9 and LGPD Article 11 protect most:
# racial or ethnic origin, religion, politics, union membership, sex life,
# genetic and biometric data. Business revenue and rent are a company's secrets.
_SENSITIVE_ATTRIBUTES = {
    DataCategory.INCOME,
    DataCategory.HEALTH,
    DataCategory.SPECIAL_CATEGORY,
    DataCategory.BUSINESS_REVENUE,
    DataCategory.RENT,
}


def sensitivity_of(category: DataCategory) -> Sensitivity:
    """Return the sensitivity level that a data category carries."""
    if category in _DIRECT_IDENTIFIERS:
        return Sensitivity.DIRECT_IDENTIFIER
    if category in _QUASI_IDENTIFIERS:
        return Sensitivity.QUASI_IDENTIFIER
    if category in _SENSITIVE_ATTRIBUTES:
        return Sensitivity.SENSITIVE
    return Sensitivity.NON_SENSITIVE
