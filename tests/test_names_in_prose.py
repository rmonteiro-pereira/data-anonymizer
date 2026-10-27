import pandas as pd
import pytest

from oblitus import Anonymizer, ProtectionLevel
from oblitus.detection import scan_column
from oblitus.detection.patterns import TEXT_RECOGNIZERS
from oblitus.detection.text_entities import find_entities, holds_entities
from oblitus.detection.text_names import find_name_and_city_spans
from oblitus.taxonomy import DataCategory
from oblitus.techniques.masking import _redact_text

NOTES = ["Atendimento de Ana Silva em Santos", "Atendimento de Maria Souza em Bauru"]


def _found(text: str) -> list[tuple[str, DataCategory]]:
    return [(text[start:end], category) for start, end, category in find_name_and_city_spans(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Atendimento de Ana Silva em Santos",
            [("Ana Silva", DataCategory.PERSON_NAME), ("Santos", DataCategory.CITY)],
        ),
        ("Ligar para Maria da Silva hoje", [("Maria da Silva", DataCategory.PERSON_NAME)]),
        (
            "Paciente José Carlos Oliveira viajou ao Rio de Janeiro",
            [
                ("José Carlos Oliveira", DataCategory.PERSON_NAME),
                ("Rio de Janeiro", DataCategory.CITY),
            ],
        ),
        ("Moved to São Paulo last year", [("São Paulo", DataCategory.CITY)]),
        ("Mora na cidade de Bauru", [("Bauru", DataCategory.CITY)]),
    ],
)
def test_names_and_cities_inside_prose_are_found(text, expected):
    assert _found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Ana ligou ontem",
        "O Santos venceu o jogo",
        "Plano em Saúde para toda a equipe",
        "Reunião de Planejamento Estratégico na segunda",
        "Relatório de Vendas de Março enviado ao Financeiro",
        "Visita à Prefeitura de Janeiro a Dezembro",
        "Rose Garden opens on Monday for the New Year",
        "The Grace Period ends during office hours",
        "Pedido de domingo entregue em Nova Embalagem",
        "Contrato com a Microsoft e o Google para Windows e Linux",
        "Esperança em dias melhores, de Segunda a Sexta",
    ],
)
def test_ordinary_prose_has_no_names_or_cities(text):
    assert _found(text) == []


def test_names_and_cities_are_redacted_with_their_labels():
    text = _redact_text(NOTES[0], TEXT_RECOGNIZERS)

    assert text == "Atendimento de <PERSON_NAME> em <CITY>"


def test_the_residue_check_sees_a_name_left_in_text():
    assert holds_entities("Atendimento de Ana Silva", TEXT_RECOGNIZERS)
    assert not holds_entities("Atendimento de <PERSON_NAME> em <CITY>", TEXT_RECOGNIZERS)


def test_names_are_dropped_when_the_categories_exclude_them():
    spans = find_entities(NOTES[0], TEXT_RECOGNIZERS, None, [DataCategory.EMAIL])

    assert spans == []


def test_a_column_of_notes_with_names_is_free_text_with_personal_data():
    finding = scan_column("payload", pd.Series(NOTES * 10))

    assert finding.category is DataCategory.FREE_TEXT_WITH_PII


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("contact", ["Ana Silva", "Maria Souza", "João da Silva", "Pedro Santos"] * 10),
        ("address", ["Rua Maria Silva 12", "Av Pedro Santos 40"] * 20),
    ],
)
def test_a_column_that_only_holds_names_is_not_taken_for_prose(column, values):
    finding = scan_column(column, pd.Series(values))

    assert finding.category is not DataCategory.FREE_TEXT_WITH_PII


def test_the_default_release_redacts_names_in_notes_and_is_not_called_anonymized():
    frame = pd.DataFrame(
        {"payload": [note for note in NOTES for _ in range(5)], "diagnosis": ["D", "H"] * 5}
    )
    anonymizer = Anonymizer(seed=1)
    plan = anonymizer.propose(frame)

    result = anonymizer.apply(frame, plan)

    assert plan.action_for("payload").technique == "redact_entities"
    released = " ".join(result.data["payload"])
    for survivor in ("Ana", "Silva", "Santos", "Maria", "Souza", "Bauru"):
        assert survivor not in released
    assert result.protection.level is not ProtectionLevel.ANONYMIZED


def test_names_left_in_the_text_make_the_release_identifiable():
    frame = pd.DataFrame({"payload": NOTES * 10, "diagnosis": ["D", "H"] * 10})
    anonymizer = Anonymizer(seed=1)
    plan = anonymizer.propose(frame)
    plan.actions = [action for action in plan.actions if action.column != "payload"] + [
        action.model_copy(update={"params": {"categories": ["email"]}})
        for action in plan.actions
        if action.column == "payload"
    ]

    result = anonymizer.apply(frame, plan)

    assert result.protection.level is ProtectionLevel.IDENTIFIABLE
    assert any("still hold recognizable personal data" in p for p in result.protection.problems)
