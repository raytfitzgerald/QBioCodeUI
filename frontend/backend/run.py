#!/usr/bin/env python3
"""Entry point for the QBioCode UI backend server."""

import sys
from pathlib import Path

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
