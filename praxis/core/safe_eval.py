"""Restricted evaluation of user-supplied mathematical expressions.

Expressions are parsed and checked against a whitelist of syntax before
being evaluated, so strings such as ``np.load.__globals__`` or
``__import__('os')`` are rejected rather than executed.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable

# Function names callable directly or as ``np.<name>``
ALLOWED_FUNCTIONS: frozenset[str] = frozenset({
    "exp", "log", "log10", "log2", "sqrt", "abs",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "power",
})

ALLOWED_CONSTANTS: frozenset[str] = frozenset({"pi", "e"})

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)


def parse_expression(expression: str, *, reserved: Iterable[str] = ()) -> tuple[Any, list[str]]:
    """Validate *expression* and return (compiled code, free variable names).

    Free variable names are identifiers that are neither whitelisted
    functions/constants nor ``np``; names in *reserved* (e.g. ``x``) are
    excluded from the returned list but still permitted.

    Raises
    ------
    ValueError
        If the expression contains anything other than arithmetic on
        numbers, variables and whitelisted functions.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression '{expression}': {exc.msg}") from None

    reserved = set(reserved)
    names: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Unsupported syntax in expression '{expression}': {type(node).__name__}"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants are allowed in '{expression}'.")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "np"
                    and (node.attr in ALLOWED_FUNCTIONS or node.attr in ALLOWED_CONSTANTS)):
                raise ValueError(
                    f"Attribute access is limited to np.<function> in '{expression}'."
                )
        if isinstance(node, ast.Call):
            if node.keywords or not (
                (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS)
                or isinstance(node.func, ast.Attribute)
            ):
                raise ValueError(f"Unsupported function call in '{expression}'.")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ValueError(f"Names may not start with '_' in '{expression}'.")
            if (node.id not in ALLOWED_FUNCTIONS and node.id not in ALLOWED_CONSTANTS
                    and node.id != "np" and node.id not in reserved
                    and node.id not in names):
                names.append(node.id)

    return compile(tree, "<expression>", "eval"), names


def evaluate(code: Any, namespace: dict[str, Any]) -> Any:
    """Evaluate code returned by :func:`parse_expression`."""
    return eval(code, {"__builtins__": {}}, namespace)
