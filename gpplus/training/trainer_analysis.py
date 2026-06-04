"""
Build and save the unified trainer analysis payload (inits, best_parameters, average_final_parameters)
from trainer.train() results. Used by the trainer (when trainer_analysis_save_path is set),
train_eval, and the tutorial so aggregation logic lives in one place.
"""
from pathlib import Path
from typing import Any, List, Optional


def build_stored_params_from_results(results: List[dict]) -> List[dict]:
    """
    Build the list of per-init records (initial, final, best_loss, best_iter) from
    trainer.train() results using callback_data. Handles FinalParameterStorageCallback,
    IterationParameterCallback, and LBFGSInnerMetricsCallbackV3.
    """
    stored_params = []
    for result in results or []:
        callback_data = result.get("callback_data") or {}
        # FinalParameterStorageCallback returns a list per result; we collect from all
        for cb_key, value in callback_data.items():
            if "FinalParameterStorage" in cb_key or "ParameterStorage" in cb_key:
                if isinstance(value, list):
                    stored_params.extend(value)
                break
    if stored_params:
        return stored_params

    # IterationParameterCallback or LBFGSInnerMetricsCallback (built-in params)
    for result in results or []:
        callback_data = result.get("callback_data") or {}
        ip = callback_data.get("IterationParameterCallback")
        lb = callback_data.get("LBFGSInnerMetricsCallbackV3")
        if ip and isinstance(ip, dict):
            records = ip.get("records") or []
            best_iter = records[-1].get("iteration") if records else result.get("best_lbfgs_iter")
            stored_params.append({
                "initial": ip.get("initial_parameters"),
                "final": ip.get("final_parameters"),
                "best_loss": result.get("loss"),
                "best_iter": best_iter,
            })
        elif lb and isinstance(lb, dict) and lb.get("initial_parameters") is not None:
            lp = lb.get("lbfgs_parameters") or []
            best_iter = lp[-1].get("iteration") if lp else None
            stored_params.append({
                "initial": lb.get("initial_parameters"),
                "final": lb.get("final_parameters"),
                "best_loss": result.get("loss"),
                "best_iter": best_iter,
            })
    return stored_params


def build_trainer_analysis_payload(
    stored_params: List[dict],
    train_results: List[dict],
    num_epochs: int,
) -> dict:
    """
    Build the unified trainer analysis payload (inits, best_parameters, average_final_parameters)
    from stored_params and train_results. Attaches lbfgs_inner_metrics, iteration_parameters,
    lbfgs_parameters per init from callback_data.
    """
    import numpy as np

    inits_data = []
    best_run_idx = None
    best_loss = float("inf")

    lbfgs_inner_by_init = {}
    iteration_params_by_init = {}
    lbfgs_parameters_by_init = {}
    for result in (train_results or []):
        init_index = result.get("init_index")
        if init_index is None:
            continue
        init_id = int(init_index) + 1
        inner = result.get("lbfgs_inner_metrics")
        if inner is not None:
            lbfgs_inner_by_init[init_id] = inner
        cb_data = result.get("callback_data") or {}
        ip_cb = cb_data.get("IterationParameterCallback")
        if ip_cb is not None:
            iteration_params_by_init[init_id] = ip_cb
        lb_cb = cb_data.get("LBFGSInnerMetricsCallbackV3")
        if lb_cb is not None and isinstance(lb_cb, dict) and "lbfgs_parameters" in lb_cb:
            lbfgs_parameters_by_init[init_id] = lb_cb["lbfgs_parameters"]

    for i, record in enumerate(stored_params):
        initial = record.get("initial") or {}
        final = record.get("final") or {}
        # Prefer explicit init id from FinalParameterStorageCallback ("run" is 1-based init label).
        # Positional i+1 is wrong when stored_params omits some inits (e.g. failed parallel runs).
        run_raw = record.get("run")
        init_id = int(run_raw) if run_raw is not None else i + 1
        run_data = {
            "init": init_id,
            "best_iter": record.get("best_iter"),
            "loss": record.get("best_loss"),
            "initial_parameters": dict(initial),
            "final_parameters": dict(final),
        }
        if record.get("best_iter") is None:
            run_data["num_epochs"] = record.get("num_epochs", num_epochs)
            run_data["best_epoch"] = record.get("best_epoch")
            run_data["epoch"] = record.get("epoch")
        if "epoch_metrics" in record:
            run_data["epoch_metrics"] = record["epoch_metrics"]
        if init_id in lbfgs_inner_by_init:
            run_data["lbfgs_inner_metrics"] = lbfgs_inner_by_init[init_id]
        if init_id in iteration_params_by_init:
            run_data["iteration_parameters"] = iteration_params_by_init[init_id]
        if init_id in lbfgs_parameters_by_init:
            run_data["lbfgs_parameters"] = lbfgs_parameters_by_init[init_id]
        for key in ("NLL", "NIS", "LOO_NLL", "KF", "MSE", "R2", "lbfgs_stop_reason"):
            if key in record:
                run_data[key] = record[key]
        inits_data.append(run_data)

        loss = record.get("best_loss")
        if loss is not None and loss < best_loss:
            best_loss = loss
            best_run_idx = i

    best_parameters = None
    if best_run_idx is not None and best_run_idx < len(stored_params):
        r = stored_params[best_run_idx]
        br_run = r.get("run")
        best_init = int(br_run) if br_run is not None else best_run_idx + 1
        best_parameters = {
            "init": best_init,
            "best_iter": r.get("best_iter"),
            "loss": r.get("best_loss"),
            "initial_parameters": dict(r.get("initial") or {}),
            "final_parameters": dict(r.get("final") or {}),
        }
        if r.get("best_iter") is None:
            best_parameters["num_epochs"] = r.get("num_epochs", num_epochs)
            best_parameters["best_epoch"] = r.get("best_epoch")
            best_parameters["epoch"] = r.get("epoch")
        for key in ("NLL", "NIS", "LOO_NLL", "KF", "MSE", "R2", "lbfgs_stop_reason"):
            if key in r:
                best_parameters[key] = r[key]
        if "epoch_metrics" in r:
            best_parameters["epoch_metrics"] = r["epoch_metrics"]
        if best_init in lbfgs_inner_by_init:
            best_parameters["lbfgs_inner_metrics"] = lbfgs_inner_by_init[best_init]
        if best_init in iteration_params_by_init:
            best_parameters["iteration_parameters"] = iteration_params_by_init[best_init]
        if best_init in lbfgs_parameters_by_init:
            best_parameters["lbfgs_parameters"] = lbfgs_parameters_by_init[best_init]

    avg_final_params = {}
    if stored_params:
        param_collections = {}
        for record in stored_params:
            final = record.get("final", {})
            for key, value in final.items():
                if key.startswith("encoder_embedding_"):
                    continue
                if key not in param_collections:
                    param_collections[key] = []
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    if value and isinstance(value[0], (list, tuple)):
                        continue
                    param_collections[key].extend([float(v) for v in value if v is not None])
                elif isinstance(value, (int, float)):
                    param_collections[key].append(float(value))
                elif hasattr(value, "item"):
                    param_collections[key].append(float(value.item()))
        for key, values in param_collections.items():
            if values:
                values_array = np.array(values)
                avg_final_params[key] = {
                    "min": float(np.min(values_array)),
                    "mean": float(np.mean(values_array)),
                    "median": float(np.median(values_array)),
                    "std": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                    "max": float(np.max(values_array)),
                    "count": int(len(values_array)),
                }

    return {
        "inits": inits_data,
        "best_parameters": best_parameters,
        "average_final_parameters": avg_final_params,
    }


def build_trainer_analysis_from_results(
    train_results: List[dict],
    num_epochs: int,
) -> Optional[dict]:
    """
    Build the full trainer analysis payload from train() results only.
    Returns None if no stored_params could be built from callback_data.
    """
    stored_params = build_stored_params_from_results(train_results)
    if not stored_params:
        return None
    return build_trainer_analysis_payload(stored_params, train_results, num_epochs)


def to_json_serializable(obj: Any) -> Any:
    """Convert tensors and numpy types to JSON-serializable Python (recursive)."""
    if hasattr(obj, "item"):
        return float(obj.item()) if obj.numel() == 1 else obj.detach().cpu().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return obj


def save_trainer_analysis(payload: dict, path: str) -> Path:
    """Serialize payload to JSON and write to path. Returns resolved path."""
    import json
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_json_serializable(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
