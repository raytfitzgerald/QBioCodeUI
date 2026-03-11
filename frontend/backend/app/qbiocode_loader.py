"""
Lazy loader for qbiocode modules.

qbiocode/__init__.py eagerly imports ALL sub-packages (learning, embeddings,
evaluation, utils, visualization, data_generation), which transitively pulls
in heavy dependencies like TensorFlow, Qiskit, ray, and grpcio. On macOS with
Python 3.9, this causes a grpcio mutex deadlock.

This module pre-registers *empty* package stubs in sys.modules for qbiocode
and every sub-package. When a router later does:

    from qbiocode.data_generation.generator import generate_data

Python finds `qbiocode` and `qbiocode.data_generation` already in sys.modules
(our stubs), skips their __init__.py files, and only loads generator.py itself.
"""

import os
import sys
import types
from pathlib import Path


def _find_qbiocode_root() -> Path:
    """Locate the qbiocode package directory."""
    # From frontend/backend/app/ -> go up 3 levels to repo root
    repo_root = Path(__file__).resolve().parents[3]
    qbc = repo_root / "qbiocode"
    if qbc.is_dir() and (qbc / "__init__.py").exists():
        return qbc
    raise FileNotFoundError(f"Cannot find qbiocode package at {qbc}")


def _collect_packages(root: Path, prefix: str = "qbiocode") -> list[tuple[str, Path]]:
    """Walk the qbiocode tree and collect all (dotted_name, directory) pairs."""
    packages = [(prefix, root)]
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            child_name = f"{prefix}.{child.name}"
            packages.append((child_name, child))
            # One level deeper (e.g. qbiocode.learning.subpkg)
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and (grandchild / "__init__.py").exists():
                    packages.append((f"{child_name}.{grandchild.name}", grandchild))
    return packages


def install_lazy_stubs():
    """
    Pre-register stub package modules for qbiocode and all its sub-packages.

    Call this ONCE at application startup, BEFORE any `from qbiocode.X import Y`
    statements execute. After this call, Python will skip all __init__.py files
    in the qbiocode tree and only load the specific .py modules requested.
    """
    try:
        qbc_root = _find_qbiocode_root()
    except FileNotFoundError:
        # qbiocode not found — nothing to do
        return

    # Ensure repo root is on sys.path
    repo_root = str(qbc_root.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    packages = _collect_packages(qbc_root)

    for dotted_name, directory in packages:
        if dotted_name in sys.modules:
            continue  # already loaded (possibly for real) — don't overwrite

        stub = types.ModuleType(dotted_name)
        stub.__path__ = [str(directory)]
        stub.__package__ = dotted_name
        stub.__file__ = str(directory / "__init__.py")
        stub.__loader__ = None
        stub.__spec__ = None
        sys.modules[dotted_name] = stub
