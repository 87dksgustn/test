import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def classification_metrics(y_true, y_pred, tp_label=1):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "tp_precision": precision_score(y_true, y_pred, pos_label=tp_label, zero_division=0),
        "tp_recall": recall_score(y_true, y_pred, pos_label=tp_label, zero_division=0),
        "tp_f1": f1_score(y_true, y_pred, pos_label=tp_label, zero_division=0),
        "confusion_matrix_labels_NoTP0_TP1": cm.tolist(),
    }

def regression_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]; yp = yp[mask]
    if len(yt) == 0:
        return {"tmax_mae": np.nan, "tmax_rmse": np.nan, "tmax_r2": np.nan}
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    if len(yt) < 2:
        r2 = np.nan
    else:
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    return {"tmax_mae": mae, "tmax_rmse": rmse, "tmax_r2": r2}

def weighted_score(metric_dict, weights):
    if not metric_dict or "error" in metric_dict:
        return float("-inf")
    return sum(float(w) * float(metric_dict.get(k, 0.0)) for k, w in weights.items())

def stable_metric_summary(fold_metrics, weights, std_penalty=0.5):
    if not fold_metrics:
        return {"error": "No fold metrics."}
    keys = ["accuracy", "tp_precision", "tp_recall", "tp_f1", "tmax_mae", "tmax_rmse", "tmax_r2"]
    out = {"fold_count": len(fold_metrics)}
    for key in keys:
        vals = np.array([m.get(key, np.nan) for m in fold_metrics], dtype=float)
        if np.isfinite(vals).any():
            out[f"{key}_mean"] = float(np.nanmean(vals))
            out[f"{key}_std"] = float(np.nanstd(vals))
        else:
            out[f"{key}_mean"] = np.nan
            out[f"{key}_std"] = np.nan
        out[key] = out[f"{key}_mean"]
    mean_score = 0.0; std_score = 0.0
    for key, w in weights.items():
        mean_score += float(w) * out.get(f"{key}_mean", 0.0)
        std_score += float(w) * out.get(f"{key}_std", 0.0)
    out["weighted_mean_score"] = float(mean_score)
    out["weighted_std_score"] = float(std_score)
    out["stable_score"] = float(mean_score - std_penalty * std_score)
    return out
