"""Compare executable ASTs with the reference implementation, ignoring docstrings."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StripDocstrings(ast.NodeTransformer):
    """Allow documentation edits while checking computational code unchanged."""

    def visit_FunctionDef(self, node):
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        return self._strip(node)

    def visit_ClassDef(self, node):
        return self._strip(node)

    def _strip(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node


def test_original_executable_functions_classes_and_constants_are_preserved():
    manifest = json.loads((ROOT / "tests/original_fingerprints.json").read_text(encoding="utf-8"))
    actual = {}
    for path in (ROOT / "src/dual_conformer_gate").glob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
            else:
                continue
            if name in manifest["nodes"]:
                assert name not in actual, f"Duplicate original definition: {name}"
                executable = StripDocstrings().visit(node)
                actual[name] = hashlib.sha256(
                    ast.dump(executable, include_attributes=False).encode()
                ).hexdigest()
    assert actual == manifest["executable_nodes"]
