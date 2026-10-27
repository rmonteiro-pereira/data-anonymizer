import pandas as pd
import pytest

from oblitus import Anonymizer, ColumnAction, Protection, ProtectionLevel
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


def _keep_redacted_copy(plan, params=None):
    """The plan with its free-text column redacted instead of dropped."""
    plan.actions = [
        action if action.column != "payload" else _redaction(params) for action in plan.actions
    ]


def _redaction(params):
    return ColumnAction(column="payload", technique="redact_entities", params=params or {})


def test_the_default_anonymization_drops_notes_instead_of_trusting_a_redaction():
    frame = pd.DataFrame(
        {"payload": [note for note in NOTES for _ in range(5)], "diagnosis": ["D", "H"] * 5}
    )
    anonymizer = Anonymizer(seed=1)
    plan = anonymizer.propose(frame)

    result = anonymizer.apply(frame, plan)

    assert plan.action_for("payload").technique == "suppress"
    assert result.data["payload"].isna().all() or (result.data["payload"] == "").all()


def test_the_default_pseudonymization_still_redacts_names_in_notes():
    frame = pd.DataFrame(
        {"payload": [note for note in NOTES for _ in range(5)], "diagnosis": ["D", "H"] * 5}
    )
    anonymizer = Anonymizer(seed=1)

    plan = anonymizer.propose(frame, protection=Protection.PSEUDONYMIZATION)

    assert plan.action_for("payload").technique == "redact_entities"


def test_a_kept_redacted_copy_hides_listed_names_but_stays_at_risk():
    frame = pd.DataFrame(
        {"payload": [note for note in NOTES for _ in range(5)], "diagnosis": ["D", "H"] * 5}
    )
    anonymizer = Anonymizer(seed=1)
    plan = anonymizer.propose(frame)
    _keep_redacted_copy(plan)

    result = anonymizer.apply(frame, plan)

    released = " ".join(result.data["payload"])
    for survivor in ("Ana", "Silva", "Santos", "Maria", "Souza", "Bauru"):
        assert survivor not in released
    assert result.protection.level is ProtectionLevel.AT_RISK
    assert any("drop the column" in problem for problem in result.protection.problems)


def test_names_left_in_the_text_make_the_release_identifiable():
    frame = pd.DataFrame({"payload": NOTES * 10, "diagnosis": ["D", "H"] * 10})
    anonymizer = Anonymizer(seed=1)
    plan = anonymizer.propose(frame)
    _keep_redacted_copy(plan, {"categories": ["email"]})

    result = anonymizer.apply(frame, plan)

    assert result.protection.level is ProtectionLevel.IDENTIFIABLE
    assert any("still hold recognizable personal data" in p for p in result.protection.problems)


@pytest.mark.parametrize(
    "text",
    [
        "Atendimento de Ana  Silva concluido ontem",
        "Atendimento de Ana\tSilva concluido ontem",
        "Atendimento de Ana   da   Silva concluido ontem",
    ],
)
def test_names_are_found_whatever_whitespace_separates_their_words(text):
    found = _found(text)

    assert [category for _, category in found] == [DataCategory.PERSON_NAME]
    assert found[0][0].startswith("Ana")
    assert found[0][0].endswith("Silva")


def test_a_city_is_found_whatever_whitespace_separates_its_words():
    assert _found("Mora no Rio  de   Janeiro") == [
        ("Rio  de   Janeiro", DataCategory.CITY),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("atendimento de ana silva concluido ontem", "ana silva"),
        ("atendimento de ana  silva concluido ontem", "ana  silva"),
        ("ligar para maria da silva hoje", "maria da silva"),
        ("paciente joao carlos souza chegou cedo", "joao carlos souza"),
        ("consulta de jose santos marcada", "jose santos"),
    ],
)
def test_lowercase_names_are_found_when_a_listed_first_name_meets_a_listed_surname(text, expected):
    assert _found(text) == [(expected, DataCategory.PERSON_NAME)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("visita para sao paulo hoje", "sao paulo"),
        ("mora em rio de janeiro desde 2020", "rio de janeiro"),
        ("mudou para belo horizonte", "belo horizonte"),
        ("mora na cidade de nova friburgo", "nova friburgo"),
    ],
)
def test_lowercase_cities_of_several_words_are_found_after_a_preposition(text, expected):
    assert _found(text) == [(expected, DataCategory.CITY)]


@pytest.mark.parametrize(
    "text",
    [
        "ana ligou ontem",
        "o cliente ligou para ana hoje",
        "silva chegou cedo",
        "reuniao com o time de vendas em santos",
        "plano de saude para toda a equipe",
        "o rio de janeiro foi lindo",
        "casa grande e senzala",
    ],
)
def test_lowercase_text_without_a_full_name_or_a_city_after_a_preposition_is_left_alone(text):
    assert _found(text) == []


@pytest.mark.parametrize(
    "template",
    [
        "atendimento de {first}  {last} concluido ontem",
        "atendimento de {first} {last} concluido ontem",
        "Atendimento de {first}   {last} concluido ontem",
    ],
)
def test_notes_with_unusual_name_formats_lose_their_names_when_redacted(template):
    people = [("Ana", "Silva"), ("Maria", "Souza"), ("Pedro", "Santos"), ("Paulo", "Lima")]
    notes = []
    for first, last in people:
        if template[0] == "a":
            first, last = first.lower(), last.lower()
        notes += [template.format(first=first, last=last)] * 5
    frame = pd.DataFrame({"payload": notes, "diagnosis": ["D", "H"] * 10})
    anonymizer = Anonymizer(seed=1)

    # Anonymization drops prose columns by default, so the redaction is checked where it
    # still runs: the pseudonymization proposal.
    plan = anonymizer.propose(frame, protection=Protection.PSEUDONYMIZATION)
    result = anonymizer.apply(frame, plan)

    released = " ".join(result.data["payload"].dropna()).lower()
    for _, last in people:
        assert last.lower() not in released


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("atendimento de ana Silva concluido ontem", "ana Silva"),
        ("atendimento de Ana silva concluido ontem", "Ana silva"),
        ("atendimento de ANA SILVA concluido ontem", "ANA SILVA"),
        ("ligar para maria da Silva hoje", "maria da Silva"),
        ("paciente joao Carlos souza chegou cedo", "joao Carlos souza"),
    ],
)
def test_names_are_found_in_any_mix_of_capitals_when_both_parts_are_listed(text, expected):
    assert _found(text) == [(expected, DataCategory.PERSON_NAME)]


@pytest.mark.parametrize(
    "text",
    [
        "atendimento de ana Foobar concluido ontem",
        "atendimento de Ana foobar concluido ontem",
        "ana ligou ontem",
        "o cliente ligou para Ana hoje",
    ],
)
def test_a_listed_first_name_with_an_unlisted_word_of_mixed_case_is_left_alone(text):
    assert _found(text) == []


def test_notes_with_a_mixed_case_name_are_free_text_with_personal_data():
    notes = [
        "atendimento de ana Silva concluido ontem",
        "atendimento de maria Silva concluido ontem",
    ]

    finding = scan_column("payload", pd.Series(notes * 10))

    assert finding.category is DataCategory.FREE_TEXT_WITH_PII


def test_a_column_of_short_labelled_names_is_a_person_name_column():
    labels = ["Paciente: Ana Silva", "Paciente: Maria Silva"]

    finding = scan_column("payload", pd.Series(labels * 10))

    assert finding.category is DataCategory.PERSON_NAME


def test_the_default_anonymization_does_not_release_short_labelled_names():
    labels = [label for label in ["Paciente: Ana Silva", "Paciente: Maria Silva"] for _ in range(5)]
    frame = pd.DataFrame({"payload": labels, "diagnosis": ["D", "H"] * 5})
    anonymizer = Anonymizer(seed=1)
    plan = anonymizer.propose(frame)

    result = anonymizer.apply(frame, plan)

    assert plan.action_for("payload").technique != "keep"
    released = " ".join(result.data["payload"].dropna().astype(str))
    assert "Silva" not in released
