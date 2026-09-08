from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path


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
