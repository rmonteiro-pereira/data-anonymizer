import importlib

import pandas as pd
import pytest

from oblitus import Anonymizer
from oblitus.detection import named_entities
from oblitus.detection.named_entities import (
    GlinerBackend,
    PresidioBackend,
    load_named_entity_backend,
)
from oblitus.detection.patterns import TEXT_RECOGNIZERS
from oblitus.detection.text_entities import find_entities
from oblitus.errors import MissingExtraError, PlanError
from oblitus.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from oblitus.taxonomy import DataCategory

TEXT = "Ana Souza mora na Rua das Flores 12, email ana@example.com"


class FakeBackend:
    """Finds the names it was told about, so no model is needed."""

    def __init__(self, entities: dict[str, DataCategory]) -> None:
        self._entities = entities
        self.calls = 0

    def find(self, text: str) -> list[tuple[int, int, DataCategory]]:
        self.calls += 1
        spans = []
        for phrase, category in self._entities.items():
            start = text.find(phrase)
            if start >= 0:
                spans.append((start, start + len(phrase), category))
        return spans


@pytest.fixture
def fake_backend(monkeypatch):
    backend = FakeBackend(
        {
            "Ana Souza": DataCategory.PERSON_NAME,
            "Rua das Flores 12": DataCategory.STREET_ADDRESS,
            "ana@example.com": DataCategory.PERSON_NAME,
        }
    )
    monkeypatch.setitem(named_entities._BACKENDS, "fake", lambda: backend)
    yield backend
    load_named_entity_backend.cache_clear()


def test_a_backend_adds_names_and_addresses_to_the_pattern_findings():
    backend = FakeBackend({"Ana Souza": DataCategory.PERSON_NAME})

    spans = find_entities(TEXT, TEXT_RECOGNIZERS, backend)

    categories = [(TEXT[span.start : span.end], span.category) for span in spans]
    assert categories == [
        ("Ana Souza", DataCategory.PERSON_NAME),
        ("ana@example.com", DataCategory.EMAIL),
    ]


def test_a_pattern_keeps_the_characters_a_backend_also_claims():
    backend = FakeBackend({"ana@example.com": DataCategory.PERSON_NAME})

    spans = find_entities(TEXT, TEXT_RECOGNIZERS, backend)

    # The bundled name list also finds "Ana Souza", so only the email's span is compared.
    categories = [(TEXT[span.start : span.end], span.category) for span in spans]
    assert ("ana@example.com", DataCategory.EMAIL) in categories
    assert ("ana@example.com", DataCategory.PERSON_NAME) not in categories


def test_named_categories_drop_what_the_backend_finds_outside_them():
    backend = FakeBackend(
        {"Ana Souza": DataCategory.PERSON_NAME, "Rua das Flores 12": DataCategory.STREET_ADDRESS}
    )

    spans = find_entities(TEXT, [], backend, {DataCategory.PERSON_NAME})

    assert [span.category for span in spans] == [DataCategory.PERSON_NAME]


def test_the_redaction_plan_uses_the_named_backend(fake_backend):
    frame = pd.DataFrame({"note": [TEXT] * 30})
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[
            ColumnAction(
                column="note", technique="redact_entities", params={"entity_model": "fake"}
            )
        ],
        direct_identifiers=["note"],
        protection=Protection.PSEUDONYMIZATION,
    )

    result = Anonymizer(seed=1).apply(frame, plan)

    assert result.data["note"].iloc[0] == "<PERSON_NAME> mora na <STREET_ADDRESS>, email <EMAIL>"
    assert fake_backend.calls > 0
    assert not any("still hold recognizable" in problem for problem in result.protection.problems)


def test_the_residue_check_uses_the_backend_too(fake_backend):
    # `redact` leaves the name in place; the check on the released text must notice it.
    frame = pd.DataFrame({"note": ["Ana Souza esteve aqui"] * 30})
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[
            ColumnAction(
                column="note",
                technique="redact_entities",
                params={"categories": ["email"], "entity_model": "fake"},
            )
        ],
        direct_identifiers=["note"],
        protection=Protection.PSEUDONYMIZATION,
    )

    result = Anonymizer(seed=1).apply(frame, plan)

    assert result.data["note"].iloc[0] == "Ana Souza esteve aqui"
    assert any("still hold recognizable" in problem for problem in result.protection.problems)


def test_an_unknown_backend_name_is_a_plan_error():
    with pytest.raises(PlanError, match="unknown named-entity backend"):
        load_named_entity_backend("nonexistent")


def test_a_missing_extra_names_the_extra(monkeypatch):
    def refuse(name: str):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", refuse)

    with pytest.raises(MissingExtraError, match=r"oblitus\[ner\]"):
        GlinerBackend().find("Ana")
    with pytest.raises(MissingExtraError, match=r"oblitus\[ner\]"):
        PresidioBackend().find("Ana")


def test_a_backend_loads_no_model_until_it_reads_a_text():
    backend = GlinerBackend()

    assert backend._model is None


def test_gliner_labels_map_to_categories():
    pytest.importorskip("gliner")

    class FakeModel:
        def predict_entities(self, text, labels, threshold):
            assert labels == ["person", "address"]
            return [{"start": 0, "end": 9, "label": "person", "text": text[:9]}]

    backend = GlinerBackend(model=FakeModel())

    assert backend.find(TEXT) == [(0, 9, DataCategory.PERSON_NAME)]


def test_presidio_entities_map_to_categories():
    presidio = pytest.importorskip("presidio_analyzer")

    class FakeAnalyzer:
        def analyze(self, text, language, entities, score_threshold):
            return [presidio.RecognizerResult("LOCATION", 13, 30, 0.9)]

    backend = PresidioBackend(analyzer=FakeAnalyzer())

    assert backend.find(TEXT) == [(13, 30, DataCategory.STREET_ADDRESS)]
