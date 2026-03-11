import asyncio
import json
import pickle
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models.schemas import SageTrainRequest, SagePredictRequest
from ..services import dataset_service
from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/sage", tags=["sage"])


def _run_sage_train(req_dict: dict, progress_callback=None):
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import pandas as pd
    from apps.sage.sage import QuantumSage

    run_id = req_dict["profiler_run_id"]
    run_dir = settings.RESULTS_DIR / "profiler" / run_id
    mr_file = run_dir / "ModelResults.csv"

    if not mr_file.exists():
        raise ValueError(f"Profiler results not found for run {run_id}")

    if progress_callback:
        progress_callback(0.1, "Loading profiler results...")

    results_df = pd.read_csv(mr_file)

    if progress_callback:
        progress_callback(0.2, "Initializing QSage...")

    sage_type = req_dict.get("sage_type", "random_forest")
    if sage_type == "rf":
        sage_type = "random_forest"

    sage = QuantumSage(results_df)
    sage.set_seed(req_dict.get("seed", 42))

    if progress_callback:
        progress_callback(0.3, f"Training {sage_type} sub-sages...")

    sage.train_sub_sages(
        test_size=req_dict.get("test_size", 0.2),
        sage_type=sage_type,
        n_iter=req_dict.get("n_iter", 50),
        cv=req_dict.get("cv", 5),
    )

    if progress_callback:
        progress_callback(0.8, "Saving trained model...")

    # Save sage model
    sage_id = str(uuid.uuid4())[:8]
    sage_dir = settings.RESULTS_DIR / "sage" / sage_id
    sage_dir.mkdir(parents=True, exist_ok=True)

    with open(sage_dir / "sage_model.pkl", "wb") as f:
        pickle.dump(sage, f)

    meta = {
        "sage_id": sage_id,
        "profiler_run_id": run_id,
        "sage_type": sage_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_size": req_dict.get("test_size", 0.2),
        "n_iter": req_dict.get("n_iter", 50),
        "cv": req_dict.get("cv", 5),
    }
    (sage_dir / "config.json").write_text(json.dumps(meta, indent=2))

    if progress_callback:
        progress_callback(1.0, "QSage training complete")

    return meta


def _run_sage_predict(req_dict: dict, progress_callback=None):
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import pandas as pd
    import numpy as np
    from qbiocode import evaluate

    sage_id = req_dict["sage_model_id"]
    sage_dir = settings.RESULTS_DIR / "sage" / sage_id

    if not (sage_dir / "sage_model.pkl").exists():
        raise ValueError(f"Sage model {sage_id} not found")

    if progress_callback:
        progress_callback(0.1, "Loading QSage model...")

    with open(sage_dir / "sage_model.pkl", "rb") as f:
        sage = pickle.load(f)

    if progress_callback:
        progress_callback(0.3, "Loading and evaluating dataset...")

    df = dataset_service.load_dataset_df(req_dict["dataset_id"])
    if df is None:
        raise ValueError("Dataset not found")

    label_col = req_dict.get("label_column", "class")
    if label_col not in df.columns:
        label_col = df.columns[-1]

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Compute complexity features
    eval_df = evaluate(X, y, "prediction_input")

    if progress_callback:
        progress_callback(0.6, "Running predictions...")

    metric = req_dict.get("metric", "f1_score")
    predictions = sage.predict(eval_df, metric=metric)

    if progress_callback:
        progress_callback(1.0, "Prediction complete")

    if isinstance(predictions, pd.DataFrame):
        return {"predictions": predictions.to_dict(orient="records")}
    return {"predictions": str(predictions)}


@router.post("/train")
async def train_sage(req: SageTrainRequest):
    job = job_manager.create_job("sage_training", req.model_dump())
    asyncio.ensure_future(job_manager.run_job(job, _run_sage_train, req.model_dump()))
    return {"job_id": job.id}


@router.post("/predict")
async def predict_sage(req: SagePredictRequest):
    job = job_manager.create_job("sage_prediction", req.model_dump())
    asyncio.ensure_future(job_manager.run_job(job, _run_sage_predict, req.model_dump()))
    return {"job_id": job.id}


@router.get("/models")
async def list_sage_models():
    sage_dir = settings.RESULTS_DIR / "sage"
    if not sage_dir.exists():
        return {"models": []}
    models = []
    for d in sage_dir.iterdir():
        if d.is_dir():
            config_file = d / "config.json"
            if config_file.exists():
                models.append(json.loads(config_file.read_text()))
    return {"models": sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)}
