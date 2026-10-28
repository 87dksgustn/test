import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from metrics_utils import classification_metrics, regression_metrics, stable_metric_summary
from models_gp import fit_gpc_passfail, fit_gpr_tmax_given_pass

def evaluate_gpc_cv(x_transformed, y_class, y_tmax=None, pass_label=0, tp_label=1, n_splits=5, weights=None, std_penalty=0.5, params=None, random_state=42, use_ard=True):
    """Evaluate GP classifier with cross-validation.
    
    Args:
        use_ard: If True, use ARD kernel (per-feature length scales).
    """
    unique, counts = np.unique(y_class, return_counts=True)
    if len(unique) < 2:
        return {"summary": {"error": "Only one class is present."}, "fold_metrics": []}
    splits = max(2, min(n_splits, int(counts.min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    for fold, (tr, va) in enumerate(cv.split(x_transformed, y_class)):
        clf = fit_gpc_passfail(x_transformed[tr], y_class[tr], random_state=random_state + fold, params=params, use_ard=use_ard)
        pred = clf.predict(x_transformed[va])
        m = classification_metrics(y_class[va], pred, tp_label=tp_label)
        if y_tmax is not None:
            reg, has = fit_gpr_tmax_given_pass(
                x_transformed[tr], y_tmax[tr], y_class[tr],
                pass_label=pass_label, min_pass_samples=4, random_state=random_state + fold, params=None, use_ard=use_ard
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


def evaluate_gp_cv_with_extra(x_transformed, y_class, y_tmax, y_extra, pass_label=0, tp_label=1, 
                               n_splits=5, weights=None, std_penalty=0.5, params=None, tmax_params=None,
                               random_state=42, use_ard=True, extra_cols=None):
    """Evaluate GP classifier + Tmax + Extra outputs with cross-validation.
    
    Returns CV metrics for classifier, tmax, and each extra output column.
    """
    unique, counts = np.unique(y_class, return_counts=True)
    if len(unique) < 2:
        return {"summary": {"error": "Only one class is present."}, "fold_metrics": [], "extra_fold_metrics": {}}
    
    splits = max(2, min(n_splits, int(counts.min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    extra_fold_metrics = {col: [] for col in (extra_cols or [])}
    
    for fold, (tr, va) in enumerate(cv.split(x_transformed, y_class)):
        # Classifier
        clf = fit_gpc_passfail(x_transformed[tr], y_class[tr], random_state=random_state + fold, params=params, use_ard=use_ard)
        pred = clf.predict(x_transformed[va])
        m = classification_metrics(y_class[va], pred, tp_label=tp_label)
        
        pass_mask_tr = (y_class[tr] == pass_label)
        pass_mask_va = (y_class[va] == pass_label)
        
        # Tmax regression
        if y_tmax is not None:
            reg, has = fit_gpr_tmax_given_pass(
                x_transformed[tr], y_tmax[tr], y_class[tr],
                pass_label=pass_label, min_pass_samples=4, random_state=random_state + fold, params=tmax_params, use_ard=use_ard
            )
            if has and int(pass_mask_va.sum()) > 0:
                tpred = reg.predict(x_transformed[va][pass_mask_va])
                m.update(regression_metrics(y_tmax[va][pass_mask_va], tpred))
            else:
                m.update({"tmax_mae": np.nan, "tmax_rmse": np.nan, "tmax_r2": np.nan})
            m["tmax_eval_n"] = int(pass_mask_va.sum())
        
        # Extra outputs regression (NoTP only)
        if y_extra is not None and extra_cols and int(pass_mask_tr.sum()) >= 4 and int(pass_mask_va.sum()) > 0:
            for i, col in enumerate(extra_cols):
                try:
                    # Fit GP regressor for this extra output on NoTP samples
                    y_col_tr = y_extra[tr, i]
                    y_col_va = y_extra[va, i]
                    
                    # Create a simple GPR for extra output
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
                    kernel = C(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
                    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=random_state + fold)
                    gpr.fit(x_transformed[tr][pass_mask_tr], y_col_tr[pass_mask_tr])
                    
                    epred = gpr.predict(x_transformed[va][pass_mask_va])
                    emetrics = regression_metrics(y_col_va[pass_mask_va], epred)
                    extra_fold_metrics[col].append({
                        "fold": fold,
                        "rmse": emetrics.get("tmax_rmse", np.nan),
                        "mae": emetrics.get("tmax_mae", np.nan),
                        "r2": emetrics.get("tmax_r2", np.nan),
                        "n_samples": int(pass_mask_va.sum()),
                    })
                except Exception:
                    extra_fold_metrics[col].append({
                        "fold": fold, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "n_samples": 0
                    })
        
        m["model"] = "gp"
        m["fold"] = fold
        fold_metrics.append(m)
    
    summary = stable_metric_summary(fold_metrics, weights or {"tp_recall": 0.7, "tp_f1": 0.3}, std_penalty)
    summary["cv_splits"] = splits
    
    # Summarize extra outputs CV
    extra_summary = {}
    for col, folds in extra_fold_metrics.items():
        if folds:
            rmses = [f["rmse"] for f in folds if np.isfinite(f["rmse"])]
            maes = [f["mae"] for f in folds if np.isfinite(f["mae"])]
            r2s = [f["r2"] for f in folds if np.isfinite(f["r2"])]
            extra_summary[col] = {
                "rmse_mean": float(np.mean(rmses)) if rmses else np.nan,
                "rmse_std": float(np.std(rmses)) if rmses else np.nan,
                "mae_mean": float(np.mean(maes)) if maes else np.nan,
                "r2_mean": float(np.mean(r2s)) if r2s else np.nan,
                "r2_std": float(np.std(r2s)) if r2s else np.nan,
                "n_folds": len(rmses),
            }
    
    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "extra_fold_metrics": extra_fold_metrics,
        "extra_summary": extra_summary,
    }

def fold_metrics_to_df(results_by_model):
    rows = []
    for model_name, result in results_by_model.items():
        for m in result.get("fold_metrics", []):
            row = dict(m); row["cv_model_group"] = model_name
            rows.append(row)
    return pd.DataFrame(rows)
