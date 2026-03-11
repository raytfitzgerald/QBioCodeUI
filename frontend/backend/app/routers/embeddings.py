import asyncio
import sys
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from ..models.schemas import EmbeddingRequest
from ..services import dataset_service
from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])

EMBEDDING_METHODS = [
    {"method": "none", "label": "None (Raw)", "description": "No dimensionality reduction", "params": []},
    {"method": "pca", "label": "PCA", "description": "Principal Component Analysis", "params": ["n_components"]},
    {"method": "nmf", "label": "NMF", "description": "Non-negative Matrix Factorization", "params": ["n_components"]},
    {"method": "lle", "label": "LLE", "description": "Locally Linear Embedding", "params": ["n_components", "n_neighbors"]},
    {"method": "isomap", "label": "Isomap", "description": "Isometric Feature Mapping", "params": ["n_components", "n_neighbors"]},
    {"method": "spectral", "label": "Spectral", "description": "Spectral Embedding", "params": ["n_components"]},
    {"method": "umap", "label": "UMAP", "description": "Uniform Manifold Approximation", "params": ["n_components", "n_neighbors"]},
    {"method": "pqk", "label": "PQK", "description": "Projected Quantum Kernel embedding", "params": ["n_components", "encoding", "entanglement", "reps"]},
]


@router.get("/methods")
async def get_methods():
    return {"methods": EMBEDDING_METHODS}


def _run_embedding(req_dict: dict, progress_callback=None):
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from qbiocode.embeddings.embed import get_embeddings

    if progress_callback:
        progress_callback(0.1, "Loading dataset...")

    df = dataset_service.load_dataset_df(req_dict["dataset_id"])
    if df is None:
        raise ValueError("Dataset not found")

    label_col = req_dict.get("label_column", "class")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    y = df[label_col]
    X = df.drop(columns=[label_col])

    qc = req_dict.get("quantum_config", {})
    seed = qc.get("seed", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.3, random_state=seed
    )

    method = req_dict.get("method", "pca")
    n_components = req_dict.get("n_components", 3)
    n_neighbors = req_dict.get("n_neighbors", 30)

    if progress_callback:
        progress_callback(0.3, f"Computing {method} embedding...")

    if method == "pqk":
        from qbiocode.embeddings.embed import pqk
        args = {
            "backend": qc.get("backend", "simulator"),
            "seed": seed,
            "shots": qc.get("shots", 1024),
            "resil_level": qc.get("resil_level", 1),
        }
        X_train_emb, X_test_emb = pqk(X_train, X_test, args)
    elif method == "none":
        X_train_emb, X_test_emb = X_train, X_test
    else:
        X_train_emb, X_test_emb = get_embeddings(
            method, X_train, X_test,
            n_components=min(n_components, X_train.shape[1]),
            n_neighbors=n_neighbors,
        )

    if progress_callback:
        progress_callback(0.8, "Saving embedded dataset...")

    # Combine and save as new dataset
    cols = [f"comp_{i}" for i in range(X_train_emb.shape[1])]
    df_train = pd.DataFrame(X_train_emb, columns=cols)
    df_train[label_col] = y_train
    df_test = pd.DataFrame(X_test_emb, columns=cols)
    df_test[label_col] = y_test
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    meta = dataset_service.get_dataset(req_dict["dataset_id"])
    orig_name = meta.get("name", "dataset") if meta else "dataset"
    new_name = f"{orig_name}_{method}_{n_components}d"

    ds_id = dataset_service.save_dataset(
        name=new_name,
        df=df_combined,
        origin="generated",
        label_column=label_col,
    )

    if progress_callback:
        progress_callback(1.0, "Embedding complete")

    # Return 2D/3D coordinates for visualization
    viz_data = None
    if X_train_emb.shape[1] <= 3:
        viz_data = {
            "coordinates": df_combined[cols].values.tolist(),
            "labels": df_combined[label_col].tolist(),
            "dimensions": len(cols),
        }

    return {"dataset_id": ds_id, "name": new_name, "visualization": viz_data}


@router.post("/compute")
async def compute_embedding(req: EmbeddingRequest):
    job = job_manager.create_job("embedding", req.model_dump())
    asyncio.ensure_future(job_manager.run_job(job, _run_embedding, req.model_dump()))
    return {"job_id": job.id}
