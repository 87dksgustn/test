from __future__ import annotations

import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from acquisition import predict_outputs


def build_config_snapshot(config_module):
    return {
        "base_continuous_cols": list(getattr(config_module, "BASE_CONTINUOUS_COLS", [])),
        "continuous_cols": list(getattr(config_module, "CONTINUOUS_COLS", [])),
        "discrete_cols": list(getattr(config_module, "DISCRETE_COLS", [])),
        "interaction_terms": [list(t) for t in getattr(config_module, "INTERACTION_TERMS", [])],
        "continuous_bounds": dict(getattr(config_module, "CONTINUOUS_BOUNDS", {})),
        "discrete_levels": dict(getattr(config_module, "DISCRETE_LEVELS", {})),
        "other_regression_cols": list(getattr(config_module, "OTHER_REGRESSION_COLS", [])),
        "time_feature_cols": list(getattr(config_module, "TIME_FEATURE_COLS", [])),
        "tp_label": int(getattr(config_module, "TP_LABEL", 1)),
        "notp_label": int(getattr(config_module, "NOTP_LABEL", 0)),
        "pass_label": int(getattr(config_module, "PASS_LABEL", 0)),
        "fail_label": int(getattr(config_module, "FAIL_LABEL", 1)),
        "model_mode": str(getattr(config_module, "MODEL_MODE", "gp")),
        "input_csv": str(getattr(config_module, "INPUT_CSV", "")),
    }


def add_interaction_terms(df, interaction_terms):
    if not interaction_terms:
        return df
    out = df.copy()
    for col1, col2, new_col in interaction_terms:
        if col1 in out.columns and col2 in out.columns:
            out[new_col] = out[col1].astype(float) * out[col2].astype(float)
    return out


def validate_prediction_inputs(df, cfg):
    missing = [c for c in cfg["base_continuous_cols"] + cfg["discrete_cols"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing input columns: " + ", ".join(missing)
            + " | Required columns are: "
            + ", ".join(cfg["base_continuous_cols"] + cfg["discrete_cols"])
        )

    for col in cfg["base_continuous_cols"]:
        df[col] = df[col].astype(float)
        if col in cfg["continuous_bounds"]:
            lo, hi = cfg["continuous_bounds"][col]
            bad = df[(df[col] < float(lo)) | (df[col] > float(hi))]
            if len(bad):
                sample_rows = [int(i) + 2 for i in bad.index[:5].tolist()]
                raise ValueError(
                    f"{col} must be within [{lo}, {hi}]. "
                    f"Found {len(bad)} out-of-range row(s). "
                    f"Example CSV row numbers: {sample_rows}"
                )

    for col in cfg["discrete_cols"]:
        allowed = set(cfg["discrete_levels"].get(col, []))
        if allowed:
            bad_vals = sorted(set(df[col].astype(str)) - allowed)
            if bad_vals:
                bad_rows = df[df[col].astype(str).isin(bad_vals)]
                sample_rows = [int(i) + 2 for i in bad_rows.index[:5].tolist()]
                raise ValueError(
                    f"{col} contains invalid level(s): {bad_vals}. "
                    f"Allowed: {sorted(allowed)}. "
                    f"Example CSV row numbers: {sample_rows}"
                )

    return df


def make_feature_frame(input_df, cfg):
    base_cols = cfg["base_continuous_cols"] + cfg["discrete_cols"]
    df = input_df.loc[:, base_cols].copy()
    df = validate_prediction_inputs(df, cfg)
    df = add_interaction_terms(df, cfg["interaction_terms"])
    return df[cfg["continuous_cols"] + cfg["discrete_cols"]]


def mask_tp_regression_outputs(df: pd.DataFrame) -> pd.DataFrame:
    if "predicted_label" not in df.columns:
        return df

    out = df.copy()
    tp_mask = out["predicted_label"].astype(str).str.upper() == "TP"
    cols_to_mask = []

    for col in ["tmax_pred", "tmax_std"]:
        if col in out.columns:
            cols_to_mask.append(col)

    cols_to_mask.extend(
        col for col in out.columns
        if col.lower() in {"max_power_pred", "max_power_std"}
    )

    if cols_to_mask:
        out.loc[tp_mask, cols_to_mask] = np.nan
    return out


def prediction_dict_to_frame(pred, cfg):
    out = pd.DataFrame({
        "p_tp": np.asarray(pred["p_tp"], dtype=float),
        "p_notp": np.asarray(pred["p_notp"], dtype=float),
        "predicted_label": np.where(np.asarray(pred["p_tp"], dtype=float) >= 0.5, "TP", "NoTP"),
        "tmax_pred": np.asarray(pred["tmax_pred"], dtype=float),
        "tmax_std": np.asarray(pred["tmax_std"], dtype=float),
    })

    if "clf_uncertainty" in pred:
        out["clf_uncertainty"] = np.asarray(pred["clf_uncertainty"], dtype=float)
    if "extra_pred" in pred and "extra_cols" in pred:
        extra_cols = list(pred["extra_cols"])
        extra_pred = np.asarray(pred["extra_pred"], dtype=float)
        extra_std = np.asarray(pred.get("extra_std"), dtype=float) if pred.get("extra_std") is not None else None
        for idx, col in enumerate(extra_cols):
            out[f"{col}_pred"] = extra_pred[:, idx]
            if extra_std is not None and extra_std.ndim == 2 and idx < extra_std.shape[1]:
                out[f"{col}_std"] = extra_std[:, idx]
    return mask_tp_regression_outputs(out)


def save_surrogate_bundle(bundle_path, preprocessor, model, config_module, metadata=None):
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessor": preprocessor,
        "model": model,
        "config": build_config_snapshot(config_module),
        "metadata": metadata or {},
    }
    with bundle_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle_path


def load_surrogate_bundle(bundle_path):
    bundle_path = Path(bundle_path)
    with bundle_path.open("rb") as f:
        return pickle.load(f)


def predict_with_bundle(bundle, input_df):
    cfg = bundle["config"]
    features = make_feature_frame(input_df, cfg)
    x = bundle["preprocessor"].transform(features)
    pred = predict_outputs(bundle["model"], x)
    return prediction_dict_to_frame(pred, cfg)


def save_bundle_metadata(metadata_path, bundle_path, bundle):
    metadata_path = Path(metadata_path)
    payload = {
        "bundle_path": str(Path(bundle_path).resolve()),
        "created_at": bundle.get("created_at"),
        "config": bundle.get("config", {}),
        "metadata": bundle.get("metadata", {}),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def get_bundle_info(bundle):
    cfg = bundle.get("config", {})
    meta = bundle.get("metadata", {})
    legacy_id_src = "|".join([
        str(bundle.get("created_at", "")),
        str(meta.get("selected_model", "")),
        str(meta.get("input_csv", cfg.get("input_csv", ""))),
    ])
    legacy_model_id = "legacy-" + hashlib.sha1(legacy_id_src.encode("utf-8")).hexdigest()[:12]
    return {
        "model_id": meta.get("model_id", legacy_model_id),
        "created_at": bundle.get("created_at"),
        "selected_model": meta.get("selected_model", "unknown"),
        "input_csv": meta.get("input_csv", cfg.get("input_csv", "")),
        "try_dir": meta.get("try_dir", ""),
        "model_mode": cfg.get("model_mode", ""),
        "base_continuous_cols": cfg.get("base_continuous_cols", []),
        "discrete_cols": cfg.get("discrete_cols", []),
        "discrete_levels": cfg.get("discrete_levels", {}),
        "continuous_bounds": cfg.get("continuous_bounds", {}),
    }
