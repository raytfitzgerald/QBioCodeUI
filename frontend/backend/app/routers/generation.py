import asyncio
import os
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter

from ..models.schemas import GenerateRequest
from ..services import dataset_service
from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/generate", tags=["generation"])

GENERATOR_TYPES = [
    {
        "type": "circles",
        "label": "Concentric Circles",
        "description": "2D concentric circles dataset for binary classification",
        "params": ["n_samples", "noise"],
    },
    {
        "type": "moons",
        "label": "Interleaving Moons",
        "description": "2D interleaving half-circles dataset",
        "params": ["n_samples", "noise"],
    },
    {
        "type": "classes",
        "label": "Classification",
        "description": "High-dimensional classification dataset with configurable features",
        "params": ["n_samples", "n_features", "n_informative", "n_redundant", "n_clusters_per_class", "weights", "n_classes"],
    },
    {
        "type": "s_curve",
        "label": "S-Curve",
        "description": "3D S-curve manifold dataset",
        "params": ["n_samples", "noise"],
    },
    {
        "type": "spheres",
        "label": "Concentric Spheres",
        "description": "N-dimensional concentric spheres dataset",
        "params": ["n_samples", "dim", "rad"],
    },
    {
        "type": "spirals",
        "label": "Intertwined Spirals",
        "description": "N-dimensional intertwined spirals dataset",
        "params": ["n_samples", "n_classes", "noise", "dim"],
    },
    {
        "type": "swiss_roll",
        "label": "Swiss Roll",
        "description": "3D Swiss roll manifold dataset",
        "params": ["n_samples", "noise", "hole"],
    },
]


@router.get("/types")
async def get_generator_types():
    return {"types": GENERATOR_TYPES}


def _run_generation(req_dict: dict, progress_callback=None):
    # Add project root to path so qbiocode is importable
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from qbiocode.data_generation.generator import generate_data
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        if progress_callback:
            progress_callback(0.1, "Generating datasets...")

        generate_data(
            type_of_data=req_dict["type"],
            save_path=tmpdir,
            n_samples=req_dict.get("n_samples", [100, 200]),
            noise=req_dict.get("noise", [0.1, 0.3]),
            hole=req_dict.get("hole", [True, False]),
            n_classes=req_dict.get("n_classes", [2]),
            dim=req_dict.get("dim", [3, 6]),
            rad=req_dict.get("rad", [3, 6]),
            n_features=req_dict.get("n_features", [10, 30]),
            n_informative=req_dict.get("n_informative", [2, 6]),
            n_redundant=req_dict.get("n_redundant", [2, 6]),
            n_clusters_per_class=req_dict.get("n_clusters_per_class", [1]),
            weights=req_dict.get("weights", [[0.5, 0.5]]),
            random_state=req_dict.get("random_state", 42),
        )

        # Register all generated CSVs as datasets
        csv_files = list(Path(tmpdir).rglob("*.csv"))
        dataset_ids = []
        total = len(csv_files) if csv_files else 1

        for i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path)
            name = f"{req_dict.get('save_name', 'generated')}_{csv_path.stem}"
            ds_id = dataset_service.save_dataset(
                name=name,
                df=df,
                origin="generated",
                filename=csv_path.name,
                label_column=df.columns[-1] if len(df.columns) > 0 else None,
            )
            dataset_ids.append(ds_id)
            if progress_callback:
                progress_callback(0.1 + 0.9 * (i + 1) / total, f"Registered {i + 1}/{total} datasets")

    return {"dataset_ids": dataset_ids, "count": len(dataset_ids)}


@router.post("")
async def generate_data(req: GenerateRequest):
    job = job_manager.create_job("data_generation", req.model_dump())
    asyncio.ensure_future(job_manager.run_job(job, _run_generation, req.model_dump()))
    return {"job_id": job.id}
