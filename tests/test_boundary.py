"""Enforces the engine/GUI boundary: ``filmneg.engine`` must never import Qt.

Two independent checks: importing the engine package must not pull Qt into
``sys.modules`` at runtime, and a static source scan must not find a Qt
import anywhere under ``engine/`` -- including inside function bodies, where
a lazy/deferred import wouldn't show up in ``sys.modules`` unless that code
path actually ran.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent / "src" / "filmneg" / "engine"
_QT_MODULE_PREFIXES = ("PySide6", "PyQt6", "PyQt5", "PySide2")


def _is_qt_module(name: str) -> bool:
    return name.startswith(_QT_MODULE_PREFIXES)


def test_engine_import_does_not_pull_in_qt():
    importlib.import_module("filmneg.engine")
    importlib.import_module("filmneg.engine.io")
    importlib.import_module("filmneg.engine.settings")

    loaded_qt_modules = [name for name in sys.modules if _is_qt_module(name)]
    assert not loaded_qt_modules, f"importing filmneg.engine pulled in Qt: {loaded_qt_modules}"


def test_engine_source_has_no_qt_imports():
    offenders = []
    for path in sorted(ENGINE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            offenders.extend(
                f"{path.relative_to(ENGINE_DIR.parent.parent.parent)}: {name}"
                for name in names
                if name and _is_qt_module(name)
            )

    assert not offenders, "Qt imports found under engine/:\n" + "\n".join(offenders)
