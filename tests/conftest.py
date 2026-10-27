import numpy as np
import pandas as pd
import pytest
from faker import Faker

from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.techniques import TechniqueContext

ROWS = 400


@pytest.fixture
def people() -> pd.DataFrame:
    """A customer table with direct identifiers, quasi-identifiers and a sensitive column."""
    rng = np.random.default_rng(7)
    faker = Faker("pt_BR")
    faker.seed_instance(7)
    ages = rng.integers(18, 80, ROWS)
    birth_dates = pd.Timestamp("2025-06-30") - pd.to_timedelta(ages * 365 + 100, unit="D")
    notes = [
        f"Ligar para {faker.email()} amanhã" if index % 3 == 0 else "sem observações"
        for index in range(ROWS)
    ]
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, ROWS + 1),
            "full_name": [faker.name() for _ in range(ROWS)],
            "email": [faker.email() for _ in range(ROWS)],
            "cpf": [faker.cpf() for _ in range(ROWS)],
            "phone": [faker.cellphone_number() for _ in range(ROWS)],
            "birth_date": birth_dates,
            "age": ages,
            "cep": [faker.postcode() for _ in range(ROWS)],
            "city": rng.choice(["Rio de Janeiro", "São Paulo", "Recife", "Curitiba"], ROWS),
            "gender": rng.choice(["M", "F"], ROWS),
            "salary": (ages * 120 + rng.normal(0, 600, ROWS)).round(2),
            "diagnosis": rng.choice(["J45", "E11", "I10", "none"], ROWS),
            "notes": notes,
            "basket_value": rng.gamma(2.0, 80.0, ROWS).round(2),
        }
    )


@pytest.fixture
def context() -> TechniqueContext:
    return TechniqueContext(
        rng=np.random.default_rng(0),
        secret_key=b"test-key",
        surrogates=SurrogateFactory("en_US"),
    )
