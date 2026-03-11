import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models.schemas import ProfilerRequest
from ..services import dataset_service
from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/profiler", tags=["profiler"])


@router.get("/config/defaults")
async def get_default_config():
    """Return default QProfiler config as JSON."""
    return {
        "models": ["svc", "dt", "lr", "nb", "rf", "mlp"],
        "embeddings": ["none", "pca"],
        "n_components": 3,
        "iterations": 2,
        "test_size": 0.3,
        "scaling": True,
        "stratify": True,
        "grid_search": False,
        "n_jobs": 4,
        "quantum_config": {
            "backend": "simulator",
            "seed": 42,
            "shots": 1024,
            "resil_level": 1,
        },
        "model_params": {
            "svc_args": {"C": 0.01, "gamma": 0.1, "kernel": "linear"},
            "dt_args": {"criterion": "gini"},
            "lr_args": {"penalty": "l2", "C": 1.0, "solver": "saga", "max_iter": 10000},
            "nb_args": {"var_smoothing": 1e-9},
            "rf_args": {"n_estimators": 100, "max_features": "sqrt"},
            "mlp_args": {"hidden_layer_sizes": "(100,)", "activation": "relu", "solver": "adam"},
            "qsvc_args": {"C": 1, "encoding": "ZZ", "entanglement": "linear", "reps": 2},
            "qnn_args": {"encoding": "Z", "entanglement": "linear", "reps": 2, "maxiter": 100, "ansatz_type": "amp"},
            "vqc_args": {"encoding": "Z", "entanglement": "linear", "reps": 2, "maxiter": 100, "ansatz_type": "amp"},
            "pqk_args": {"encoding": "Z", "entanglement": "linear", "reps": 2},
        },
    }


def _run_profiler(req_dict: dict, progress_callback=None):
    """Replicate qprofiler.py main loop without Hydra."""
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
    from qbiocode import evaluate, get_embeddings, model_run

    dataset_ids = req_dict["dataset_ids"]
    models = req_dict.get("models", ["svc", "dt", "lr", "nb", "rf"])
    embeddings = req_dict.get("embeddings", ["none"])
    n_components = req_dict.get("n_components", 3)
    iterations = req_dict.get("iterations", 2)
    test_size = req_dict.get("test_size", 0.3)
    do_scaling = req_dict.get("scaling", True)
    stratify = req_dict.get("stratify", True)
    grid_search = req_dict.get("grid_search", False)
    n_jobs = req_dict.get("n_jobs", 4)
    qc = req_dict.get("quantum_config", {})
    model_params = req_dict.get("model_params", {})

    seed = qc.get("seed", 42)
    np.random.seed(seed)

    # Create results directory
    run_id = str(uuid.uuid4())[:8]
    run_dir = settings.RESULTS_DIR / "profiler" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    total_steps = len(dataset_ids) * iterations * len(embeddings)
    current_step = 0

    all_model_results = []
    all_eval_results = []

    args = {
        "backend": qc.get("backend", "simulator"),
        "seed": seed,
        "q_seed": qc.get("q_seed", 42),
        "shots": qc.get("shots", 1024),
        "resil_level": qc.get("resil_level", 1),
        "grid_search": grid_search,
        "cross_validation": req_dict.get("cross_validation", 5),
        "model": models,
        "n_jobs": n_jobs,
    }

    # Add model-specific args
    for key, val in model_params.items():
        args[key] = val

    for ds_idx, ds_id in enumerate(dataset_ids):
        if progress_callback:
            progress_callback(
                current_step / max(total_steps, 1),
                f"Processing dataset {ds_idx + 1}/{len(dataset_ids)}"
            )

        df = dataset_service.load_dataset_df(ds_id)
        if df is None:
            continue

        meta = dataset_service.get_dataset(ds_id)
        ds_name = meta.get("name", ds_id) if meta else ds_id
        label_col = meta.get("label_column", df.columns[-1]) if meta else df.columns[-1]

        if label_col not in df.columns:
            label_col = df.columns[-1]

        y = df[label_col].values
        X = df.drop(columns=[label_col])

        # Encode labels
        if not np.issubdtype(type(y[0]), np.number):
            enc = OrdinalEncoder()
            y = enc.fit_transform(y.reshape(-1, 1)).ravel()

        # Raw data evaluation
        try:
            eval_df = evaluate(X, y, ds_name)
            eval_dict = {}
            for col in eval_df.columns:
                val = eval_df[col].iloc[0] if len(eval_df) > 0 else None
                try:
                    eval_dict[col] = float(val) if val is not None else None
                except (ValueError, TypeError):
                    eval_dict[col] = str(val) if val is not None else None
            eval_dict["Dataset"] = ds_name
            eval_dict["embeddings"] = "raw"
            all_eval_results.append(eval_dict)
        except Exception as e:
            print(f"Evaluation failed for {ds_name}: {e}")

        for iteration in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y, test_size=test_size, random_state=seed + iteration,
                stratify=y if stratify else None
            )

            if do_scaling:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            for emb_name in embeddings:
                current_step += 1
                if progress_callback:
                    progress_callback(
                        current_step / max(total_steps, 1),
                        f"Dataset: {ds_name} | Iter: {iteration + 1}/{iterations} | Embedding: {emb_name}"
                    )

                if emb_name == "none":
                    X_tr_emb, X_te_emb = X_train, X_test
                else:
                    try:
                        nc = min(n_components, X_train.shape[1])
                        X_tr_emb, X_te_emb = get_embeddings(emb_name, X_train, X_test, n_components=nc)
                    except Exception as e:
                        print(f"Embedding {emb_name} failed: {e}")
                        continue

                # Run models
                try:
                    results = model_run(
                        pd.DataFrame(X_tr_emb),
                        pd.DataFrame(X_te_emb),
                        pd.Series(y_train),
                        pd.Series(y_test),
                        ds_name,
                        args
                    )
                    if results:
                        for model_key, model_result in results.items():
                            if isinstance(model_result, pd.DataFrame):
                                result_cols = [c for c in model_result.columns if c.startswith("results_")]
                                for rc in result_cols:
                                    res = model_result[rc].iloc[0]
                                    if isinstance(res, str):
                                        import ast
                                        res = ast.literal_eval(res)
                                    if isinstance(res, dict):
                                        row = {
                                            "Dataset": ds_name,
                                            "embeddings": emb_name,
                                            "iteration": iteration,
                                            "model": res.get("model", model_key),
                                            "accuracy": res.get("accuracy", 0),
                                            "f1_score": res.get("f1_score", 0),
                                            "auc": res.get("auc", None),
                                            "time": res.get("time", 0),
                                        }
                                        all_model_results.append(row)
                except Exception as e:
                    print(f"model_run failed for {ds_name}/{emb_name}/{iteration}: {e}")

    # Save results
    if all_model_results:
        pd.DataFrame(all_model_results).to_csv(run_dir / "ModelResults.csv", index=False)
    if all_eval_results:
        pd.DataFrame(all_eval_results).to_csv(run_dir / "RawDataEvaluation.csv", index=False)

    # Save config
    config_out = {
        "run_id": run_id,
        "dataset_ids": dataset_ids,
        "models": models,
        "embeddings": embeddings,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "n_model_results": len(all_model_results),
        "n_eval_results": len(all_eval_results),
    }
    (run_dir / "config.json").write_text(json.dumps(config_out, indent=2))

    if progress_callback:
        progress_callback(1.0, "Profiler run complete")

    return config_out


@router.post("/run")
async def run_profiler(req: ProfilerRequest):
    job = job_manager.create_job("profiler_run", req.model_dump())
    asyncio.ensure_future(job_manager.run_job(job, _run_profiler, req.model_dump()))
    return {"job_id": job.id}


@router.get("/runs")
async def list_runs():
    profiler_dir = settings.RESULTS_DIR / "profiler"
    if not profiler_dir.exists():
        return {"runs": []}
    runs = []
    for d in profiler_dir.iterdir():
        if d.is_dir():
            config_file = d / "config.json"
            if config_file.exists():
                config = json.loads(config_file.read_text())
                runs.append(config)
    return {"runs": sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run_dir = settings.RESULTS_DIR / "profiler" / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")

    config = {}
    config_file = run_dir / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())

    model_results = []
    mr_file = run_dir / "ModelResults.csv"
    if mr_file.exists():
        import pandas as pd
        df = pd.read_csv(mr_file)
        model_results = df.to_dict(orient="records")

    eval_results = []
    er_file = run_dir / "RawDataEvaluation.csv"
    if er_file.exists():
        import pandas as pd
        df = pd.read_csv(er_file)
        eval_results = df.to_dict(orient="records")

    return {
        "config": config,
        "model_results": model_results,
        "eval_results": eval_results,
    }
