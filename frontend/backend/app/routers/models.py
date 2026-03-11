import asyncio
import sys
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from ..models.schemas import TrainRequest
from ..services import dataset_service
from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/models", tags=["models"])

MODEL_SCHEMAS = {
    "svc": {
        "label": "Support Vector Classifier",
        "category": "classical",
        "params": {
            "C": {"type": "float", "default": 1.0, "description": "Regularization parameter"},
            "kernel": {"type": "select", "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"},
            "gamma": {"type": "string", "default": "scale", "description": "Kernel coefficient"},
            "degree": {"type": "int", "default": 3, "description": "Degree for poly kernel"},
        },
    },
    "dt": {
        "label": "Decision Tree",
        "category": "classical",
        "params": {
            "criterion": {"type": "select", "options": ["gini", "entropy", "log_loss"], "default": "gini"},
            "max_depth": {"type": "int", "default": None, "description": "Maximum tree depth"},
            "min_samples_split": {"type": "int", "default": 2},
            "min_samples_leaf": {"type": "int", "default": 1},
        },
    },
    "nb": {
        "label": "Naive Bayes",
        "category": "classical",
        "params": {
            "var_smoothing": {"type": "float", "default": 1e-9, "description": "Variance smoothing"},
        },
    },
    "lr": {
        "label": "Logistic Regression",
        "category": "classical",
        "params": {
            "penalty": {"type": "select", "options": ["l2", "l1", "elasticnet", "none"], "default": "l2"},
            "C": {"type": "float", "default": 1.0},
            "solver": {"type": "select", "options": ["saga", "lbfgs", "liblinear", "newton-cg"], "default": "saga"},
            "max_iter": {"type": "int", "default": 10000},
        },
    },
    "rf": {
        "label": "Random Forest",
        "category": "classical",
        "params": {
            "n_estimators": {"type": "int", "default": 100},
            "criterion": {"type": "select", "options": ["gini", "entropy", "log_loss"], "default": "gini"},
            "max_depth": {"type": "int", "default": None},
            "min_samples_split": {"type": "int", "default": 2},
            "min_samples_leaf": {"type": "int", "default": 1},
            "max_features": {"type": "select", "options": ["sqrt", "log2", "None"], "default": "sqrt"},
            "bootstrap": {"type": "bool", "default": True},
        },
    },
    "xgb": {
        "label": "XGBoost",
        "category": "classical",
        "params": {
            "n_estimators": {"type": "int", "default": 100},
            "max_depth": {"type": "int", "default": 6},
            "learning_rate": {"type": "float", "default": 0.5},
            "subsample": {"type": "float", "default": 0.5},
            "colsample_bytree": {"type": "float", "default": 1.0},
            "min_child_weight": {"type": "int", "default": 1},
        },
    },
    "mlp": {
        "label": "Multi-Layer Perceptron",
        "category": "classical",
        "params": {
            "hidden_layer_sizes": {"type": "string", "default": "100", "description": "Comma-separated layer sizes"},
            "activation": {"type": "select", "options": ["relu", "tanh", "logistic", "identity"], "default": "relu"},
            "solver": {"type": "select", "options": ["adam", "sgd", "lbfgs"], "default": "adam"},
            "alpha": {"type": "float", "default": 0.0001},
            "max_iter": {"type": "int", "default": 10000},
            "learning_rate": {"type": "select", "options": ["constant", "adaptive", "invscaling"], "default": "constant"},
        },
    },
    "qsvc": {
        "label": "Quantum SVC",
        "category": "quantum",
        "params": {
            "C": {"type": "float", "default": 1.0},
            "encoding": {"type": "select", "options": ["ZZ", "Z", "P"], "default": "ZZ"},
            "entanglement": {"type": "select", "options": ["linear", "circular", "full"], "default": "linear"},
            "reps": {"type": "int", "default": 2},
            "primitive": {"type": "select", "options": ["sampler", "estimator"], "default": "sampler"},
            "pegasos": {"type": "bool", "default": False},
        },
    },
    "qnn": {
        "label": "Quantum Neural Network",
        "category": "quantum",
        "params": {
            "primitive": {"type": "select", "options": ["sampler", "estimator"], "default": "sampler"},
            "local_optimizer": {"type": "select", "options": ["COBYLA", "L_BFGS_B", "GradientDescent"], "default": "COBYLA"},
            "maxiter": {"type": "int", "default": 100},
            "encoding": {"type": "select", "options": ["Z", "ZZ", "P"], "default": "Z"},
            "entanglement": {"type": "select", "options": ["linear", "circular", "full"], "default": "linear"},
            "reps": {"type": "int", "default": 2},
            "ansatz_type": {"type": "select", "options": ["amp", "esu2", "twolocal"], "default": "amp"},
        },
    },
    "vqc": {
        "label": "Variational Quantum Classifier",
        "category": "quantum",
        "params": {
            "primitive": {"type": "select", "options": ["sampler", "estimator"], "default": "sampler"},
            "local_optimizer": {"type": "select", "options": ["COBYLA", "L_BFGS_B", "GradientDescent"], "default": "COBYLA"},
            "maxiter": {"type": "int", "default": 100},
            "encoding": {"type": "select", "options": ["Z", "ZZ", "P"], "default": "Z"},
            "entanglement": {"type": "select", "options": ["linear", "circular", "full"], "default": "linear"},
            "reps": {"type": "int", "default": 2},
            "ansatz_type": {"type": "select", "options": ["amp", "esu2", "twolocal"], "default": "amp"},
        },
    },
    "pqk": {
        "label": "Projected Quantum Kernel",
        "category": "quantum",
        "params": {
            "encoding": {"type": "select", "options": ["Z", "ZZ", "P"], "default": "Z"},
            "entanglement": {"type": "select", "options": ["linear", "circular", "full"], "default": "linear"},
            "reps": {"type": "int", "default": 2},
            "primitive": {"type": "select", "options": ["estimator", "sampler"], "default": "estimator"},
        },
    },
}


@router.get("/available")
async def get_available_models():
    return {"models": MODEL_SCHEMAS}


def _run_training(req_dict: dict, progress_callback=None):
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from qbiocode.utils.helper_fn import scaler_fn, feature_encoding
    from qbiocode.learning.compute_svc import compute_svc, compute_svc_opt
    from qbiocode.learning.compute_dt import compute_dt, compute_dt_opt
    from qbiocode.learning.compute_nb import compute_nb, compute_nb_opt
    from qbiocode.learning.compute_lr import compute_lr, compute_lr_opt
    from qbiocode.learning.compute_rf import compute_rf, compute_rf_opt
    from qbiocode.learning.compute_mlp import compute_mlp, compute_mlp_opt
    from qbiocode.learning.compute_qsvc import compute_qsvc
    from qbiocode.learning.compute_qnn import compute_qnn
    from qbiocode.learning.compute_vqc import compute_vqc
    from qbiocode.learning.compute_pqk import compute_pqk

    if progress_callback:
        progress_callback(0.1, "Loading dataset...")

    df = dataset_service.load_dataset_df(req_dict["dataset_id"])
    if df is None:
        raise ValueError("Dataset not found")

    label_col = req_dict.get("label_column", "class")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    y = df[label_col].values
    X = df.drop(columns=[label_col]).values

    # Encode labels if categorical
    from sklearn.preprocessing import OrdinalEncoder as OE
    if not np.issubdtype(type(y[0]), np.number):
        enc = OE()
        y = enc.fit_transform(y.reshape(-1, 1)).ravel()

    if progress_callback:
        progress_callback(0.2, "Splitting data...")

    qc = req_dict.get("quantum_config", {})
    seed = qc.get("seed", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req_dict.get("test_size", 0.3), random_state=seed, stratify=y
    )

    # Scale
    scaling = req_dict.get("scaling", "MinMaxScaler")
    if scaling == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        s = MinMaxScaler()
        X_train = s.fit_transform(X_train)
        X_test = s.transform(X_test)
    elif scaling == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler()
        X_train = s.fit_transform(X_train)
        X_test = s.transform(X_test)

    if progress_callback:
        progress_callback(0.3, f"Training {req_dict['model']}...")

    # Build args dict
    args = {
        "backend": qc.get("backend", "simulator"),
        "seed": seed,
        "q_seed": qc.get("q_seed", 42),
        "shots": qc.get("shots", 1024),
        "resil_level": qc.get("resil_level", 1),
        "grid_search": req_dict.get("grid_search", False),
        "cross_validation": req_dict.get("cross_validation", 5),
    }

    model_name = req_dict["model"]
    params = req_dict.get("params", {})
    grid_search = req_dict.get("grid_search", False)

    compute_fns = {
        "svc": (compute_svc, compute_svc_opt),
        "dt": (compute_dt, compute_dt_opt),
        "nb": (compute_nb, compute_nb_opt),
        "lr": (compute_lr, compute_lr_opt),
        "rf": (compute_rf, compute_rf_opt),
        "mlp": (compute_mlp, compute_mlp_opt),
        "qsvc": (compute_qsvc, None),
        "qnn": (compute_qnn, None),
        "vqc": (compute_vqc, None),
        "pqk": (compute_pqk, None),
    }

    if model_name == "xgb":
        try:
            from qbiocode.learning.compute_xgb import compute_xgb, compute_xgb_opt
            compute_fns["xgb"] = (compute_xgb, compute_xgb_opt)
        except ImportError:
            raise ValueError("XGBoost not available")

    if model_name not in compute_fns:
        raise ValueError(f"Unknown model: {model_name}")

    fn_standard, fn_opt = compute_fns[model_name]

    if grid_search and fn_opt:
        result_df = fn_opt(X_train, X_test, y_train, y_test, args, **params)
    else:
        result_df = fn_standard(X_train, X_test, y_train, y_test, args, **params)

    if progress_callback:
        progress_callback(1.0, "Training complete")

    # Extract results
    result_cols = [c for c in result_df.columns if c.startswith("results_")]
    if result_cols:
        res = result_df[result_cols[0]].iloc[0]
        if isinstance(res, str):
            import ast
            res = ast.literal_eval(res)
        return {
            "model": model_name,
            "accuracy": res.get("accuracy", 0),
            "f1_score": res.get("f1_score", 0),
            "auc": res.get("auc", None),
            "time": res.get("time", 0),
            "params": res.get("Model_Parameters", res.get("BestParams_GridSearch", {})),
        }

    return {"model": model_name, "error": "No results produced"}


@router.post("/train")
async def train_model(req: TrainRequest):
    job = job_manager.create_job("model_training", req.model_dump())
    asyncio.ensure_future(job_manager.run_job(job, _run_training, req.model_dump()))
    return {"job_id": job.id}
