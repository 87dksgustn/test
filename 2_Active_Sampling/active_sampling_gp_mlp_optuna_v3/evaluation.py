import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from metrics_utils import classification_metrics, regression_metrics, stable_metric_summary
from models_gp import fit_gpc_passfail, fit_gpr_tmax_given_pass

def evaluate_gpc_cv(x_transformed, y_class, y_tmax=None, pass_label=0, tp_label=1, n_splits=5, weights=None, std_penalty=0.5, params=None, random_state=42):
    unique, counts = np.unique(y_class, return_counts=True)
    if len(unique) < 2:
        return {"summary": {"error": "Only one class is present."}, "fold_metrics": []}
    splits = max(2, min(n_splits, int(counts.min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    for fold, (tr, va) in enumerate(cv.split(x_transformed, y_class)):
        clf = fit_gpc_passfail(x_transformed[tr], y_class[tr], random_state=random_state + fold, params=params)
        pred = clf.predict(x_transformed[va])
        m = classification_metrics(y_class[va], pred, tp_label=tp_label)
        if y_tmax is not None:
            reg, has = fit_gpr_tmax_given_pass(
                x_transformed[tr], y_tmax[tr], y_class[tr],
                pass_label=pass_label, min_pass_samples=4, random_state=random_state + fold, params=None
            )
            pass_mask = (y_class[va] == pass_label)
            if has and int(pass_mask.sum()) > 0:
                tpred = reg.predict(x_transformed[va][pass_mask])
                m.update(regression_metrics(y_tmax[va][pass_mask], tpred))
            else:
                m.update({"tmax_mae": np.nan, "tmax_rmse": np.nan, "tmax_r2": np.nan})
            m["tmax_eval_n"] = int(pass_mask.sum())
        m["model"] = "gp"; m["fold"] = fold
        fold_metrics.append(m)
    summary = stable_metric_summary(fold_metrics, weights or {"tp_recall":0.7,"tp_f1":0.3}, std_penalty)
    summary["cv_splits"] = splits
    return {"summary": summary, "fold_metrics": fold_metrics}

def fold_metrics_to_df(results_by_model):
    rows = []
    for model_name, result in results_by_model.items():
        for m in result.get("fold_metrics", []):
            row = dict(m); row["cv_model_group"] = model_name
            rows.append(row)
    return pd.DataFrame(rows)
