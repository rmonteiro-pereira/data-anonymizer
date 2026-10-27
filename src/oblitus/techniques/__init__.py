"""The catalogue of masking, pseudonymization, generalization, noise and perturbation techniques.

A technique is a function `(values, context, **params) -> values`. To add your own,
build a `Technique` and pass it to `Anonymizer(extra_techniques=[...])`.
"""

from oblitus.techniques.catalogue import (
    BUILTIN_TECHNIQUES,
    StatisticalEffect,
    Technique,
    TechniqueFamily,
)
from oblitus.techniques.context import TechniqueContext

__all__ = [
    "BUILTIN_TECHNIQUES",
    "StatisticalEffect",
    "Technique",
    "TechniqueContext",
    "TechniqueFamily",
]
