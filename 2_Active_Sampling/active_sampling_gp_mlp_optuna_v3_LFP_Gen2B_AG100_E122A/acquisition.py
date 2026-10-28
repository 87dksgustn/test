import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from models_mlp import predict_mlp_ensemble

def safe_minmax_scale(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0: return v
    mn = np.nanmin(v); mx = np.nanmax(v)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx-mn) < 1e-12:
        return np.zeros_like(v)
    return (v-mn)/(mx-mn)

def p_window_score(p_notp, low=0.60, high=0.90, center=0.75):
    p = np.asarray(p_notp, dtype=float); s = np.zeros_like(p)
    left = (p >= low) & (p <= center); right = (p > center) & (p <= high)
    if center > low: s[left] = (p[left]-low)/(center-low)
    if high > center: s[right] = (high-p[right])/(high-center)
    return np.clip(s, 0, 1)

def compute_local_sparsity(x_candidate, x_train):
    nn = NearestNeighbors(n_neighbors=1).fit(x_train)
    dist, _ = nn.kneighbors(x_candidate)
    return safe_minmax_scale(dist.ravel())

def compute_misclass_repair_score(candidate_df, misclass_df, base_cols, bounds, combo_col="discrete_combo_id", length_scale=0.15, same_combo_only=True):
    """Score candidates by proximity to previous holdout misclassified points.

    score = max over misclassified points of exp(-normalized_dist / length_scale),
    optionally restricted to candidates sharing the same discrete combo.
    """
    n = len(candidate_df)
    scores = np.zeros(n, dtype=float)
    if misclass_df is None or len(misclass_df) == 0:
        return scores
    cols = [c for c in base_cols if c in misclass_df.columns and c in candidate_df.columns]
    if not cols:
        return scores
    lo = np.array([bounds[c][0] for c in cols], dtype=float)
    span = np.array([max(bounds[c][1] - bounds[c][0], 1e-12) for c in cols], dtype=float)
    cand = (candidate_df[cols].to_numpy(dtype=float) - lo) / span
    mis = (misclass_df[cols].to_numpy(dtype=float) - lo) / span
    cand_combo = candidate_df[combo_col].to_numpy() if combo_col in candidate_df.columns else None
    mis_combo = misclass_df[combo_col].to_numpy() if combo_col in misclass_df.columns else None
    for j in range(len(mis)):
        d = np.linalg.norm(cand - mis[j], axis=1)
        s = np.exp(-d / max(float(length_scale), 1e-12))
        if same_combo_only and cand_combo is not None and mis_combo is not None:
            s = np.where(cand_combo == mis_combo[j], s, 0.0)
        scores = np.maximum(scores, s)
    return scores

def compute_combo_priority(candidate_df, labeled_df, combo_col, min_samples_per_combo, max_samples_per_combo):
    counts = labeled_df[combo_col].value_counts().to_dict(); out = []
    for cid in candidate_df[combo_col].values:
        n = counts.get(cid, 0)
        if n >= max_samples_per_combo: p = 0.0
        elif n < min_samples_per_combo: p = 1.0
        else: p = (max_samples_per_combo - n) / max(1, max_samples_per_combo - min_samples_per_combo)
        out.append(p)
    return np.asarray(out, dtype=float)

def predict_gp_outputs(models, x):
    proba = models.clf.predict_proba(x)
    tp_idx = list(models.clf.classes_).index(1)
    p_tp = proba[:, tp_idx]
    p_notp = 1 - p_tp
    boundary = np.clip(1 - 2*np.abs(p_tp-0.5), 0, 1)
    clf_unc = np.zeros_like(boundary)
    if getattr(models, "clf_ensemble", None):
        tp_samples = []
        for clf_i in models.clf_ensemble:
            p_i = clf_i.predict_proba(x)
            tp_i = p_i[:, list(clf_i.classes_).index(1)]
            tp_samples.append(tp_i)
        if tp_samples:
            clf_unc = np.std(np.vstack(tp_samples), axis=0)
    if models.has_tmax_model:
        tpred, tstd = models.reg_tmax.predict(x, return_std=True)
    else:
        tpred = np.zeros(x.shape[0]); tstd = np.zeros(x.shape[0])
    return {"p_tp": p_tp, "p_notp": p_notp, "boundary_score": boundary, "clf_uncertainty": clf_unc, "tmax_pred": tpred, "tmax_std": tstd}


def predict_hybrid_outputs(models, x):
    """Predict outputs using HybridModels bundle (different model per task)."""
    n = x.shape[0]
    
    # Classifier prediction
    if models.clf_kind == "mlp":
        clf_pred = predict_mlp_ensemble(models.clf_model, x)
        p_tp = clf_pred["p_tp"]
        clf_unc = clf_pred["p_tp_std"]
    else:
        clf_pred = predict_gp_outputs(models.clf_model, x)
        p_tp = clf_pred["p_tp"]
        clf_unc = clf_pred["clf_uncertainty"]
    
    p_notp = 1 - p_tp
    boundary = np.clip(1 - 2 * np.abs(p_tp - 0.5), 0, 1)
    
    # Tmax regressor prediction
    if models.tmax_kind == "mlp":
        tmax_pred_result = predict_mlp_ensemble(models.tmax_model, x)
        tmax_pred = tmax_pred_result["tmax_pred"]
        tmax_std = tmax_pred_result["tmax_std"]
    else:
        if models.tmax_model.has_tmax_model:
            tmax_pred, tmax_std = models.tmax_model.reg_tmax.predict(x, return_std=True)
        else:
            tmax_pred = np.zeros(n)
            tmax_std = np.zeros(n)
    
    result = {
        "p_tp": p_tp,
        "p_notp": p_notp,
        "boundary_score": boundary,
        "clf_uncertainty": clf_unc,
        "tmax_pred": tmax_pred,
        "tmax_std": tmax_std,
    }
    
    # Extra outputs prediction (each can use different model)
    if models.extra_cols and models.extra_models:
        extra_preds = []
        extra_stds = []
        for col in models.extra_cols:
            if col in models.extra_models:
                model, kind = models.extra_models[col]
                if kind == "mlp":
                    epred = predict_mlp_ensemble(model, x)
                    if "extra_pred" in epred and epred.get("extra_cols"):
                        col_idx = list(epred["extra_cols"]).index(col) if col in epred["extra_cols"] else -1
                        if col_idx >= 0:
                            extra_preds.append(epred["extra_pred"][:, col_idx])
                            extra_stds.append(epred.get("extra_std", np.zeros((n, len(epred["extra_cols"]))))[:, col_idx])
                        else:
                            extra_preds.append(np.zeros(n))
                            extra_stds.append(np.zeros(n))
                    else:
                        extra_preds.append(np.zeros(n))
                        extra_stds.append(np.zeros(n))
                else:
                    # GP model
                    if hasattr(model, "reg_extra") and col in model.reg_extra:
                        ep, estd = model.reg_extra[col].predict(x, return_std=True)
                        extra_preds.append(ep)
                        extra_stds.append(estd)
                    else:
                        extra_preds.append(np.zeros(n))
                        extra_stds.append(np.zeros(n))
        
        if extra_preds:
            result["extra_pred"] = np.column_stack(extra_preds)
            result["extra_std"] = np.column_stack(extra_stds)
            result["extra_cols"] = models.extra_cols
    
    return result


def predict_outputs(models, x):
    # Hybrid model: route to dedicated function
    if getattr(models, "kind", "gp") == "hybrid":
        return predict_hybrid_outputs(models, x)
    
    if getattr(models, "kind", "gp") == "mlp":
        pred = predict_mlp_ensemble(models, x)
        p_tp = pred["p_tp"]
        boundary = np.clip(1 - 2*np.abs(p_tp-0.5), 0, 1)
        result = {
            "p_tp": p_tp,
            "p_notp": pred["p_notp"],
            "boundary_score": boundary,
            "clf_uncertainty": pred["p_tp_std"],
            "tmax_pred": pred["tmax_pred"],
            "tmax_std": pred["tmax_std"],
        }
        # Include extra outputs if available (for reporting only)
        if "extra_pred" in pred and "extra_cols" in pred:
            result["extra_pred"] = pred["extra_pred"]
            result["extra_std"] = pred.get("extra_std")
            result["extra_cols"] = pred["extra_cols"]
        return result
    # GP model
    gp_result = predict_gp_outputs(models, x)
    # Add extra outputs from GP regressors if available
    if hasattr(models, "reg_extra") and models.reg_extra and models.extra_cols:
        extra_preds = []
        extra_stds = []
        for col in models.extra_cols:
            if col in models.reg_extra:
                ep, estd = models.reg_extra[col].predict(x, return_std=True)
                extra_preds.append(ep)
                extra_stds.append(estd)
        if extra_preds:
            gp_result["extra_pred"] = np.column_stack(extra_preds)
            gp_result["extra_std"] = np.column_stack(extra_stds)
            gp_result["extra_cols"] = models.extra_cols
    return gp_result

def compute_acquisition_scores(candidate_df, labeled_df, x_candidate_transformed, x_train_transformed, models, config):
    pred = predict_outputs(models, x_candidate_transformed)
    local = compute_local_sparsity(x_candidate_transformed, x_train_transformed)
    combo = compute_combo_priority(candidate_df, labeled_df, "discrete_combo_id", config.MIN_SAMPLES_PER_COMBO, config.MAX_SAMPLES_PER_COMBO)
    tmax_scaled = safe_minmax_scale(pred["tmax_pred"])
    tmax_unc = safe_minmax_scale(pred["tmax_std"])
    clf_unc = safe_minmax_scale(pred["clf_uncertainty"])
    notp_win = p_window_score(pred["p_notp"], config.NOTP_WINDOW_LOW, config.NOTP_WINDOW_HIGH, config.NOTP_WINDOW_CENTER)
    bw = config.BOUNDARY_WEIGHTS_MLP if getattr(models, "kind", "gp") == "mlp" else config.BOUNDARY_WEIGHTS_GP
    tw = config.NOTP_HIGH_TMAX_WEIGHTS; uw = config.UNCERTAINTY_SPARSE_WEIGHTS
    acq_b = bw["boundary"]*pred["boundary_score"] + bw["clf_uncertainty"]*clf_unc + bw["local_sparsity"]*local + bw["combo_priority"]*combo
    acq_t = tw["tmax"]*tmax_scaled + tw["notp_window"]*notp_win + tw["tmax_uncertainty"]*tmax_unc + tw["local_sparsity"]*local + tw["combo_priority"]*combo
    acq_u = uw["clf_uncertainty"]*clf_unc + uw["tmax_uncertainty"]*tmax_unc + uw["local_sparsity"]*local + uw["combo_priority"]*combo
    out = candidate_df.copy()
    out["selected_model_kind"] = getattr(models, "kind", "gp")
    out["p_tp"] = pred["p_tp"]; out["p_notp"] = pred["p_notp"]; out["boundary_score"] = pred["boundary_score"]
    out["clf_uncertainty_raw"] = pred["clf_uncertainty"]; out["clf_uncertainty_scaled"] = clf_unc
    out["tmax_pred_given_notp"] = pred["tmax_pred"]; out["tmax_std_given_notp"] = pred["tmax_std"]
    out["tmax_scaled"] = tmax_scaled; out["notp_window_score"] = notp_win; out["local_sparsity"] = local; out["combo_priority"] = combo
    out["acq_boundary"] = acq_b; out["acq_notp_high_tmax"] = acq_t; out["acq_uncertainty_sparse"] = acq_u
    # Add extra output predictions (view-only, not used for scoring/bucket allocation)
    if "extra_pred" in pred and "extra_cols" in pred:
        extra_cols = pred["extra_cols"]
        extra_pred = pred["extra_pred"]
        extra_std = pred.get("extra_std")
        for i, col in enumerate(extra_cols):
            out[f"{col}_pred_given_notp"] = extra_pred[:, i]
            if extra_std is not None:
                out[f"{col}_std_given_notp"] = extra_std[:, i]
    return out
