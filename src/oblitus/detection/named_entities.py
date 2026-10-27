"""Names and addresses inside free text, found by a named-entity model.

Patterns cannot tell "Ana Souza" or "Rua das Flores 12" from other words. A model can, so
free text may be read by one when the user asks for it. Two backends are supported:
- `gliner`: GLiNER with `urchade/gliner_multi_pii-v1`, a multilingual PII model;
- `presidio`: Microsoft Presidio's analyzer.
Both come from the `ner` extra (`pip install "oblitus[ner]"`). A backend loads its model when
it first reads a text, never when the library is imported.

A backend is any object with a `find(text)` method that returns `(start, end, category)`
triples, so tests and callers can pass their own.
"""

import importlib
from collections.abc import Callable
from functools import cache
from typing import Any, Protocol

from oblitus.errors import MissingExtraError, PlanError
from oblitus.taxonomy import DataCategory

EntitySpan = tuple[int, int, DataCategory]

GLINER_MODEL = "urchade/gliner_multi_pii-v1"
GLINER_THRESHOLD = 0.5
PRESIDIO_THRESHOLD = 0.4

_GLINER_LABELS = {
    "person": DataCategory.PERSON_NAME,
    "address": DataCategory.STREET_ADDRESS,
}
_PRESIDIO_ENTITIES = {
    "PERSON": DataCategory.PERSON_NAME,
    "LOCATION": DataCategory.STREET_ADDRESS,
}


class NamedEntityBackend(Protocol):
    """Anything that can point at names and addresses in a text."""

    def find(self, text: str) -> list[EntitySpan]:
        """Return the start, end and category of every entity in the text."""
        ...


class GlinerBackend:
    """Finds people and addresses with a GLiNER model."""

    def __init__(
        self,
        model_name: str = GLINER_MODEL,
        threshold: float = GLINER_THRESHOLD,
        model: Any = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._model = model

    def find(self, text: str) -> list[EntitySpan]:
        found = self._loaded_model().predict_entities(
            text, list(_GLINER_LABELS), threshold=self._threshold
        )
        return [(item["start"], item["end"], _GLINER_LABELS[item["label"]]) for item in found]

    def _loaded_model(self) -> Any:
        if self._model is None:
            gliner = _import_optional("gliner")
            self._model = gliner.GLiNER.from_pretrained(self._model_name)
        return self._model


class PresidioBackend:
    """Finds people and locations with Presidio's analyzer."""

    def __init__(
        self, language: str = "en", threshold: float = PRESIDIO_THRESHOLD, analyzer: Any = None
    ) -> None:
        self._language = language
        self._threshold = threshold
        self._analyzer = analyzer

    def find(self, text: str) -> list[EntitySpan]:
        results = self._loaded_analyzer().analyze(
            text=text,
            language=self._language,
            entities=list(_PRESIDIO_ENTITIES),
            score_threshold=self._threshold,
        )
        return [(item.start, item.end, _PRESIDIO_ENTITIES[item.entity_type]) for item in results]

    def _loaded_analyzer(self) -> Any:
        if self._analyzer is None:
            engine = _import_optional("presidio_analyzer")
            self._analyzer = engine.AnalyzerEngine()
        return self._analyzer


_BACKENDS: dict[str, Callable[[], NamedEntityBackend]] = {
    "gliner": GlinerBackend,
    "presidio": PresidioBackend,
}


@cache
def load_named_entity_backend(name: str) -> NamedEntityBackend:
    """Return the backend with this name; the same object every time, so its model loads once."""
    try:
        factory = _BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(_BACKENDS))
        raise PlanError(f"unknown named-entity backend {name!r}; use one of: {known}") from None
    return factory()


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise MissingExtraError(
            f"the {module_name} package is needed for named-entity detection; "
            'install it with: pip install "oblitus[ner]"'
        ) from error
