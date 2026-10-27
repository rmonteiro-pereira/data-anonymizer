"""Run techniques still written for pandas columns on an in-memory table.

Techniques are moving to Narwhals expressions, which run on every engine. Until each
one has moved, this bridge hands it the column as a pandas Series, with the original
table for grouping, and puts the result back. It only works in memory; on a lazy
table such a technique raises a clear error.
"""

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import narwhals as nw
import pandas as pd
import polars as pl

from data_anonymizer.techniques.context import TechniqueContext

if TYPE_CHECKING:
    from data_anonymizer.table import Table

PandasTechnique = Callable[..., pd.Series]


def pandas_column(function: PandasTechnique) -> Callable[..., nw.Expr]:
    """Adapt a `(values, context, **params) -> Series` technique to the table protocol."""

    @functools.wraps(function)
    def run(table: "Table", column: str, context: TechniqueContext, **params: Any) -> nw.Expr:
        if table.engine.is_lazy:
            raise ValueError(f"{function.__name__} runs on in-memory tables only for now")
        original = table.original_pandas()
        values = original[column]
        bridged = TechniqueContext(
            rng=context.rng,
            secret_key=context.secret_key,
            surrogates=context.surrogates,
            table=original,
            randomness_key=context.randomness_key,
        )
        result = function(values, bridged, **params)
        return table.attach(_to_narwhals_series(result, values))

    run.__signature__ = _table_signature(function)  # type: ignore[attr-defined]
    return run


def _to_narwhals_series(result: pd.Series, values: pd.Series) -> nw.Series[Any]:
    try:
        converted = pl.from_pandas(result.reset_index(drop=True))
    except (TypeError, ValueError, pl.exceptions.PolarsError):
        # Mixed Python objects have no single column type; text holds them all.
        texts = [None if pd.isna(item) else str(item) for item in result.to_numpy(dtype=object)]
        converted = pl.Series(str(values.name), texts, dtype=pl.String)
    return nw.from_native(converted, series_only=True)


def _table_signature(function: PandasTechnique) -> inspect.Signature:
    # Plans are checked by binding their parameters to this signature, so it lists
    # the table protocol's three positional inputs and the technique's own keywords.
    keywords = [
        parameter
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    positional = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ("table", "column", "context")
    ]
    return inspect.Signature([*positional, *keywords])
