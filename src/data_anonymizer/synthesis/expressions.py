"""A small, safe expression language for schema formulas, conditions and constraints.

Expressions are Python syntax restricted to arithmetic, comparisons, `and`/`or`/`not`,
`in [...]`, column names, numbers, strings and a fixed list of functions. They are
parsed with `ast` and evaluated node by node, never with `eval`, so a schema file
from someone else cannot run code.

Formulas may also call `bernoulli(p)`, `normal(mean, std)`, `poisson(rate)` and
`uniform(low, high)`, which draw one random value per row. Each call turns a stored uniform
number into its value through the inverse distribution function, so a build can be replayed
(see `draw_replay`).

Examples:
    50 + 2.5 * age
    bernoulli(sigmoid(-2 + 0.8 * support_tickets))
    where(plan == 'pro', 1.4, 1.0) * spend
    100 + 20 * sin(2 * pi * dayofyear(signup) / 365)
    end_date >= start_date
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from data_anonymizer.synthesis.draw_replay import normal_from_uniform, poisson_from_uniform

_NANOSECONDS_PER_DAY = 86_400 * 10**9

# Limits that keep a schema from someone else from using unbounded time or memory.
MAX_EXPRESSION_LENGTH = 2000
MAX_EXPRESSION_DEPTH = 100

Functions = dict[str, Callable[..., Any]]


def _date_part(part: str) -> Callable[[Any], Any]:
    def extract(values: Any) -> Any:
        return getattr(pd.Series(pd.to_datetime(values)).dt, part).to_numpy()

    return extract


def _days(values: Any) -> Any:
    dates = pd.Series(pd.to_datetime(values)).astype("datetime64[ns]")
    return dates.astype("int64").to_numpy() / _NANOSECONDS_PER_DAY


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "year": _date_part("year"),
    "month": _date_part("month"),
    "day": _date_part("day"),
    "hour": _date_part("hour"),
    "dayofweek": _date_part("day_of_week"),
    "dayofyear": _date_part("day_of_year"),
    "days": _days,
    "sigmoid": lambda values: 1 / (1 + np.exp(-np.asarray(values, dtype=float))),
}

# Functions that draw random values. They need a generator, so they are allowed in
# formulas but not in conditions or constraints, which must give the same answer
# every time they are checked.
RANDOM_FUNCTIONS = frozenset({"bernoulli", "normal", "poisson", "uniform"})


# Returns the stored uniform numbers for the next random function call of a formula.
UniformSource = Callable[[], np.ndarray]


def _fresh_uniforms(rng: np.random.Generator, rows: int) -> UniformSource:
    return lambda: rng.random(rows)


def _random_functions(next_uniform: UniformSource, rows: int) -> dict[str, Callable[..., Any]]:
    def sized(values: Any) -> np.ndarray:
        return np.broadcast_to(np.asarray(values, dtype=float), (rows,))

    def bernoulli(probability: Any) -> np.ndarray:
        return next_uniform() < sized(probability)

    def normal(mean: Any, std: Any) -> np.ndarray:
        spread = sized(std)
        if np.any(spread < 0):
            raise ValueError("normal needs a std that is not negative")
        return sized(mean) + spread * normal_from_uniform(next_uniform())

    def poisson(rate: Any) -> np.ndarray:
        return poisson_from_uniform(next_uniform(), sized(rate))

    def uniform(low: Any, high: Any) -> np.ndarray:
        return sized(low) + (sized(high) - sized(low)) * next_uniform()

    return {"bernoulli": bernoulli, "normal": normal, "poisson": poisson, "uniform": uniform}


CONSTANTS = {"pi": np.pi, "e": np.e}


def _is_plain_number(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating)


def _is_text_like(value: Any) -> bool:
    if isinstance(value, str | list | tuple):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "US":
            return True
        return value.dtype.kind == "O" and any(isinstance(item, str) for item in value)
    return False


def _power(base: Any, exponent: Any) -> Any:
    # Python integers grow without limit, so powers of plain numbers run in float64.
    if _is_plain_number(base) and _is_plain_number(exponent):
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.float64(base) ** np.float64(exponent)
        if not np.isfinite(result):
            raise ExpressionError(f"{base} ** {exponent} is too large")
        return result
    return operator.pow(base, exponent)


def _multiply(left: Any, right: Any) -> Any:
    if _is_text_like(left) or _is_text_like(right):
        raise ExpressionError("* does not repeat text or lists")
    return operator.mul(left, right)


def _remainder(left: Any, right: Any) -> Any:
    if _is_text_like(left):
        raise ExpressionError("% does not format text")
    return operator.mod(left, right)


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: _multiply,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: _remainder,
    ast.Pow: _power,
}

_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class ExpressionError(ValueError):
    """An expression is not allowed, or cannot be evaluated."""


def parse(text: str) -> ast.Expression:
    """Parse an expression and check that it only uses allowed syntax."""
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"expression has {len(text)} characters; the limit is {MAX_EXPRESSION_LENGTH}"
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"cannot parse {text!r}: {error.msg}") from error
    except (RecursionError, MemoryError, ValueError) as error:
        raise ExpressionError(f"cannot parse {text!r}: it is too deeply nested") from error
    if _depth(tree) > MAX_EXPRESSION_DEPTH:
        raise ExpressionError(
            f"{text[:40]!r}... is too deeply nested; the limit is {MAX_EXPRESSION_DEPTH} levels"
        )
    for node in ast.walk(tree):
        _check_node(node, text)
    return tree


def _depth(tree: ast.AST) -> int:
    # Counted with a stack because a recursive count would fail on the input it guards against.
    deepest = 0
    pending = [(tree, 1)]
    while pending:
        node, level = pending.pop()
        deepest = max(deepest, level)
        pending.extend((child, level + 1) for child in ast.iter_child_nodes(node))
    return deepest


def referenced_columns(text: str) -> set[str]:
    """Return the column names an expression reads."""
    tree = parse(text)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return names - called - set(CONSTANTS)


def uses_randomness(text: str) -> bool:
    """True when an expression calls a function that draws random values."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in RANDOM_FUNCTIONS
        for node in ast.walk(parse(text))
    )


def evaluate(
    text: str,
    frame: pd.DataFrame,
    rng: np.random.Generator | None = None,
    *,
    uniforms: UniformSource | None = None,
) -> np.ndarray:
    """Evaluate an expression over every row of a table and return one value per row.

    The random functions need either `rng`, which draws new numbers, or `uniforms`, which
    returns the stored numbers of the next call site, one per row.
    """
    tree = parse(text)
    functions = dict(FUNCTIONS)
    if uniforms is None and rng is not None:
        uniforms = _fresh_uniforms(rng, len(frame))
    if uniforms is not None:
        functions.update(_random_functions(uniforms, len(frame)))
    try:
        result = _evaluate(tree.body, frame, functions)
    except ExpressionError:
        raise
    except (ArithmeticError, TypeError, ValueError, RecursionError, MemoryError) as error:
        raise ExpressionError(
            f"cannot evaluate {text!r}: {type(error).__name__}: {error}"
        ) from error
    if np.ndim(result) == 0:
        return np.full(len(frame), result)
    return np.asarray(result)


_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.In,
    ast.NotIn,
    *_BINARY,
    *_COMPARISONS,
)


def _check_node(node: ast.AST, text: str) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise ExpressionError(f"{type(node).__name__} is not allowed in {text!r}")
    if isinstance(node, ast.Call):
        known = FUNCTIONS.keys() | RANDOM_FUNCTIONS
        if not isinstance(node.func, ast.Name) or node.func.id not in known:
            allowed = ", ".join(sorted(known))
            raise ExpressionError(f"unknown function in {text!r}; allowed: {allowed}")
        if node.keywords:
            raise ExpressionError(f"keyword arguments are not allowed in {text!r}")
    if isinstance(node, ast.Constant) and not isinstance(node.value, int | float | str | bool):
        raise ExpressionError(f"only numbers, strings and booleans are allowed in {text!r}")


def _evaluate(node: ast.AST, frame: pd.DataFrame, functions: Functions) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _name(node.id, frame)
    if isinstance(node, ast.List | ast.Tuple):
        return [_evaluate(element, frame, functions) for element in node.elts]
    if isinstance(node, ast.UnaryOp):
        return _unary(node, frame, functions)
    if isinstance(node, ast.BinOp):
        operation = _BINARY[type(node.op)]
        return operation(
            _evaluate(node.left, frame, functions), _evaluate(node.right, frame, functions)
        )
    if isinstance(node, ast.BoolOp):
        return _boolean(node, frame, functions)
    if isinstance(node, ast.Compare):
        return _compare(node, frame, functions)
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        name = node.func.id
        if name not in functions:
            raise ExpressionError(f"{name} draws random values and is only allowed in formulas")
        arguments = [_evaluate(argument, frame, functions) for argument in node.args]
        return functions[name](*arguments)
    raise ExpressionError(f"{type(node).__name__} is not allowed")


def _name(name: str, frame: pd.DataFrame) -> Any:
    if name in CONSTANTS:
        return CONSTANTS[name]
    if name not in frame.columns:
        raise ExpressionError(f"unknown column {name!r}")
    return frame[name].to_numpy()


def _unary(node: ast.UnaryOp, frame: pd.DataFrame, functions: Functions) -> Any:
    operand = _evaluate(node.operand, frame, functions)
    if isinstance(node.op, ast.USub):
        return -operand
    if isinstance(node.op, ast.Not):
        return np.logical_not(operand)
    return operand


def _boolean(node: ast.BoolOp, frame: pd.DataFrame, functions: Functions) -> Any:
    values = [_evaluate(value, frame, functions) for value in node.values]
    combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
    result = values[0]
    for value in values[1:]:
        result = combine(result, value)
    return result


def _compare(node: ast.Compare, frame: pd.DataFrame, functions: Functions) -> Any:
    left = _evaluate(node.left, frame, functions)
    result: Any = True
    for operation, comparator in zip(node.ops, node.comparators, strict=True):
        right = _evaluate(comparator, frame, functions)
        if isinstance(operation, ast.In | ast.NotIn):
            outcome = np.isin(left, right)
            if isinstance(operation, ast.NotIn):
                outcome = np.logical_not(outcome)
        else:
            outcome = _COMPARISONS[type(operation)](left, right)
        result = np.logical_and(result, outcome)
        left = right
    return result
