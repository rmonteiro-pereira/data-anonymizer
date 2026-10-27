"""Put the rows of a released table in a fresh random order.

A release that keeps the input order, or the input index, lines up with the original and
with every other release of the same table, row by row. A fresh order removes that join.
The order follows the run's choice key, so an anonymization run (which uses a throwaway
key) gives a different order every time, and a pseudonymization run is reproducible for
whoever holds the secret key.
"""

from oblitus.engines import Engine, WorkingFrame, add_keyed_uniforms, collect
from oblitus.table import ROW_KEY_COLUMN
from oblitus.techniques.context import TechniqueContext

_DRAW = "__da_shuffle_draw"


def shuffle_rows(
    released: WorkingFrame, engine: Engine, context: TechniqueContext
) -> tuple[WorkingFrame, list[int] | None]:
    """Return the released rows in a random order, and the old position of each new row.

    The frame must carry `ROW_KEY_COLUMN`. The positions are None for a lazy table, whose
    rows have no fixed position; pandas needs them to move its untouched columns too.
    """
    if engine.is_lazy:
        key = context.choice_key(ROW_KEY_COLUMN, "shuffle_rows")
        drawn = add_keyed_uniforms(released, engine, key=key, source=ROW_KEY_COLUMN, names=[_DRAW])
        return drawn.sort(_DRAW).drop(_DRAW), None
    rows = collect(released)
    positions = context.generator(ROW_KEY_COLUMN, "shuffle_rows").permutation(len(rows)).tolist()
    return rows[positions], positions
