from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..config import settings


INDEX_FILE = settings.DATASETS_DIR / "index.json"


def _load_index() -> dict[str, dict]:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {}


def _save_index(index: dict[str, dict]):
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def list_datasets() -> list[dict]:
    index = _load_index()
    datasets = []
    for ds_id, meta in index.items():
        meta["id"] = ds_id
        datasets.append(meta)
    return sorted(datasets, key=lambda d: d.get("created_at", ""), reverse=True)


def get_dataset(dataset_id: str) -> dict | None:
    index = _load_index()
    meta = index.get(dataset_id)
    if not meta:
        return None
    meta["id"] = dataset_id
    return meta


def get_dataset_path(dataset_id: str) -> Path | None:
    ds_dir = settings.DATASETS_DIR / dataset_id
    csv_files = list(ds_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return None


def load_dataset_df(dataset_id: str) -> pd.DataFrame | None:
    path = get_dataset_path(dataset_id)
    if path and path.exists():
        return pd.read_csv(path)
    return None


def get_dataset_preview(dataset_id: str, n_rows: int = 20) -> dict | None:
    meta = get_dataset(dataset_id)
    if not meta:
        return None
    df = load_dataset_df(dataset_id)
    if df is None:
        return None
    subset = df if n_rows == 0 else df.head(n_rows)
    return {
        "meta": meta,
        "head": subset.to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "stats": json.loads(df.describe(include="all").to_json()),
    }


def save_dataset(
    name: str,
    df: pd.DataFrame,
    origin: str = "uploaded",
    filename: str | None = None,
    label_column: str | None = None,
) -> str:
    ds_id = str(uuid.uuid4())[:8]
    ds_dir = settings.DATASETS_DIR / ds_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"{name}.csv"
    csv_path = ds_dir / filename
    df.to_csv(csv_path, index=False)

    meta = {
        "name": name,
        "filename": filename,
        "origin": origin,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_column": label_column,
    }

    index = _load_index()
    index[ds_id] = meta
    _save_index(index)

    return ds_id


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    df = pd.read_csv(pd.io.common.BytesIO(file_content))
    name = Path(filename).stem
    # Try to guess label column
    label_col = None
    for candidate in ["class", "label", "target", "y"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None and len(df.columns) > 0:
        label_col = df.columns[-1]
    return save_dataset(name=name, df=df, origin="uploaded", filename=filename, label_column=label_col)


def delete_dataset(dataset_id: str) -> bool:
    index = _load_index()
    if dataset_id not in index:
        return False
    del index[dataset_id]
    _save_index(index)
    ds_dir = settings.DATASETS_DIR / dataset_id
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    return True
