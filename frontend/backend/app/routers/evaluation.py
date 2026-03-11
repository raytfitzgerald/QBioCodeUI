import asyncio
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..models.schemas import EvaluationRequest
from ..services import dataset_service
from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])

METRICS_INFO = [
    {"key": "# Features", "label": "Feature Count", "description": "Number of features in the dataset"},
    {"key": "# Samples", "label": "Sample Count", "description": "Number of samples in the dataset"},
    {"key": "Feature_Samples_ratio", "label": "Feature/Sample Ratio", "description": "Ratio of features to samples"},
    {"key": "Intrinsic_Dimension", "label": "Intrinsic Dimension", "description": "Estimated intrinsic dimensionality"},
    {"key": "Condition number", "label": "Condition Number", "description": "Condition number of the feature matrix"},
    {"key": "Fisher Discriminant Ratio", "label": "Fisher Discriminant Ratio", "description": "Measure of class separability"},
    {"key": "Total Correlations", "label": "Total Correlations", "description": "Sum of feature correlations"},
    {"key": "Mutual information", "label": "Mutual Information", "description": "Mutual information between features and labels"},
    {"key": "# Non-zero entries", "label": "Non-zero Entries", "description": "Count of non-zero feature values"},
    {"key": "# Low variance features", "label": "Low Variance Features", "description": "Features with near-zero variance"},
    {"key": "Variation", "label": "Variation", "description": "Average feature variance"},
    {"key": "std_var", "label": "Variance Std Dev", "description": "Standard deviation of feature variances"},
    {"key": "Coefficient of Variation %", "label": "Coefficient of Variation", "description": "Average coefficient of variation (%)"},
    {"key": "std_co_of_v", "label": "CoV Std Dev", "description": "Standard deviation of coefficient of variation"},
    {"key": "Skewness", "label": "Skewness", "description": "Average feature skewness"},
    {"key": "std_skew", "label": "Skewness Std Dev", "description": "Standard deviation of skewness"},
    {"key": "Kurtosis", "label": "Kurtosis", "description": "Average feature kurtosis"},
    {"key": "std_kurt", "label": "Kurtosis Std Dev", "description": "Standard deviation of kurtosis"},
    {"key": "Mean Log Kernel Density", "label": "Mean Log Kernel Density", "description": "Average log kernel density estimate"},
    {"key": "Isomap Reconstruction Error", "label": "Isomap Recon. Error", "description": "Reconstruction error from Isomap embedding"},
    {"key": "Fractal dimension", "label": "Fractal Dimension", "description": "Higuchi fractal dimension estimate"},
    {"key": "Entropy", "label": "Entropy", "description": "Average label entropy"},
    {"key": "std_entropy", "label": "Entropy Std Dev", "description": "Standard deviation of entropy"},
]


@router.get("/metrics")
async def get_available_metrics():
    return {"metrics": METRICS_INFO}


def _run_evaluation(dataset_id: str, label_column: str, progress_callback=None):
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import pandas as pd
    from qbiocode.evaluation.dataset_evaluation import evaluate

    if progress_callback:
        progress_callback(0.1, "Loading dataset...")

    df = dataset_service.load_dataset_df(dataset_id)
    if df is None:
        raise ValueError("Dataset not found")

    meta = dataset_service.get_dataset(dataset_id)
    name = meta.get("name", dataset_id) if meta else dataset_id

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")

    y = df[label_column]
    X = df.drop(columns=[label_column])

    if progress_callback:
        progress_callback(0.3, "Computing complexity metrics...")

    result_df = evaluate(X, y, name)

    if progress_callback:
        progress_callback(1.0, "Evaluation complete")

    # Convert to dict
    metrics = {}
    for col in result_df.columns:
        val = result_df[col].iloc[0] if len(result_df) > 0 else None
        try:
            metrics[col] = float(val) if val is not None else None
        except (ValueError, TypeError):
            metrics[col] = str(val) if val is not None else None

    return {"dataset": name, "metrics": metrics}


@router.post("/complexity")
async def evaluate_complexity(req: EvaluationRequest):
    # For small datasets, run synchronously; for large ones, use a job
    df = dataset_service.load_dataset_df(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")

    if len(df) < 5000:
        try:
            result = _run_evaluation(req.dataset_id, req.label_column)
            return result
        except ValueError as e:
            raise HTTPException(400, str(e))
    else:
        job = job_manager.create_job("evaluation", req.model_dump())
        asyncio.ensure_future(
            job_manager.run_job(job, _run_evaluation, req.dataset_id, req.label_column)
        )
        return {"job_id": job.id}
