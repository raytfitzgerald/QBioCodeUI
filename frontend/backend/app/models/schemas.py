from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---- Dataset ----

class DatasetMeta(BaseModel):
    id: str
    name: str
    filename: str
    origin: Literal["uploaded", "generated"]
    rows: int
    columns: int
    column_names: list[str]
    created_at: str
    label_column: Optional[str] = None


class DatasetPreview(BaseModel):
    meta: DatasetMeta
    head: list[dict[str, Any]]
    dtypes: dict[str, str]


class DatasetListResponse(BaseModel):
    datasets: list[DatasetMeta]


# ---- Data Generation ----

class GeneratorParamSchema(BaseModel):
    name: str
    type: str
    description: str
    default: Any = None


class GeneratorType(BaseModel):
    type: str
    label: str
    description: str
    params: list[GeneratorParamSchema]


class GenerateRequest(BaseModel):
    type: Literal["circles", "moons", "classes", "s_curve", "spheres", "spirals", "swiss_roll"]
    n_samples: list[int] = Field(default=[100, 200])
    noise: list[float] = Field(default=[0.1, 0.3])
    hole: list[bool] = Field(default=[True, False])
    n_classes: list[int] = Field(default=[2])
    dim: list[int] = Field(default=[3, 6])
    rad: list[float] = Field(default=[3, 6])
    n_features: list[int] = Field(default=[10, 30])
    n_informative: list[int] = Field(default=[2, 6])
    n_redundant: list[int] = Field(default=[2, 6])
    n_clusters_per_class: list[int] = Field(default=[1])
    weights: list[list[float]] = Field(default=[[0.5, 0.5]])
    random_state: int = 42
    save_name: str = "generated"


# ---- Evaluation ----

class EvaluationRequest(BaseModel):
    dataset_id: str
    label_column: str = "class"


class EvaluationResult(BaseModel):
    dataset: str
    metrics: dict[str, Any]


# ---- Quantum Config ----

class QuantumConfig(BaseModel):
    backend: str = "simulator"
    seed: int = 42
    q_seed: int = 42
    shots: int = 1024
    resil_level: int = 1


# ---- Model Training ----

class TrainRequest(BaseModel):
    dataset_id: str
    model: Literal["svc", "dt", "nb", "lr", "rf", "xgb", "mlp", "qsvc", "qnn", "vqc", "pqk"]
    label_column: str = "class"
    test_size: float = 0.3
    scaling: Literal["None", "StandardScaler", "MinMaxScaler"] = "MinMaxScaler"
    grid_search: bool = False
    cross_validation: int = 5
    params: dict[str, Any] = Field(default_factory=dict)
    quantum_config: QuantumConfig = Field(default_factory=QuantumConfig)


class TrainResult(BaseModel):
    model: str
    accuracy: float
    f1_score: float
    auc: Optional[float] = None
    time: float
    params: dict[str, Any]


# ---- Embeddings ----

class EmbeddingRequest(BaseModel):
    dataset_id: str
    method: Literal["pca", "nmf", "lle", "isomap", "spectral", "umap", "pqk", "none"]
    label_column: str = "class"
    n_components: int = 3
    n_neighbors: int = 30
    quantum_config: QuantumConfig = Field(default_factory=QuantumConfig)


# ---- QProfiler ----

class ProfilerRequest(BaseModel):
    dataset_ids: list[str]
    models: list[str] = Field(default=["svc", "dt", "lr", "nb", "rf", "mlp"])
    embeddings: list[str] = Field(default=["none"])
    n_components: int = 3
    iterations: int = 2
    test_size: float = 0.3
    scaling: bool = True
    stratify: bool = True
    grid_search: bool = False
    n_jobs: int = 4
    quantum_config: QuantumConfig = Field(default_factory=QuantumConfig)
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ProfilerRunMeta(BaseModel):
    id: str
    created_at: str
    status: str
    dataset_count: int
    model_count: int


# ---- QSage ----

class SageTrainRequest(BaseModel):
    profiler_run_id: str
    sage_type: Literal["random_forest", "rf", "mlp"] = "random_forest"
    test_size: float = 0.2
    n_iter: int = 50
    cv: int = 5
    seed: int = 42


class SagePredictRequest(BaseModel):
    sage_model_id: str
    dataset_id: str
    label_column: str = "class"
    metric: Literal["f1_score", "accuracy", "auc"] = "f1_score"


# ---- Jobs ----

class JobResponse(BaseModel):
    id: str
    type: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: float = 0.0
    message: str = ""
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
