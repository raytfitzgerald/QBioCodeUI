#!/usr/bin/env python3
"""Entry point for the QBioCode UI backend server."""

import os
import sys
from pathlib import Path

# ── Environment fixes (MUST be before any other imports) ──────────────
# Prevent grpcio mutex deadlock on macOS with fork-based reloaders
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# Suppress noisy TF/grpc logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
# Headless matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure the project root is on the path so qbiocode is importable
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent / "app")],
    )
