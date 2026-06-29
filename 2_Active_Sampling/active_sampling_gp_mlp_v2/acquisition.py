import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from models_mlp import predict_mlp_ensemble


def safe_minmax_scale(values):
    v = np.asarray(values, dtype=float)
    mn, mx = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx-mn) < 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v-mn)/(mx-mn)


def p_window_score(p_pass, low=0.60, high=0.90, center=0.75):
    p = np.asarray(p_pass, dtype=float); s = np.zeros_like(p)
    left = (p >= low) & (p <= center); right = (p > center) & (p <= high)
    if center > low: s[left] = (p[left]-low)/(center-low)
    if high > center: s[right] = (high-p[right])/(high-center)
    return np.clip(s, 0, 1)


def compute_local_sparsity(x_candidate_transformed, x_train_transformed):
    nn = NearestNeighbors(n_neighbors=1).fit(x_train_transformed)
    d,_ = nn.kneighbors(x_candidate_transformed)
    return safe_minmax_scale(d.ravel())


def compute_combo_priority(candidate_df, labeled_df, combo_col, min_samples_per_combo, max_samples_per_combo):
    counts = labeled_df[combo_col].value_counts().to_dict()
    out=[]
    for cid in candidate_df[combo_col].values:
        n = counts.get(cid, 0)
        if n >= max_samples_per_combo: p=0.0
        elif n < min_samples_per_combo: p=1.0
        else: p=(max_samples_per_combo-n)/max(1, max_samples_per_combo-min_samples_per_combo)
        out.append(p)
    return np.asarray(out, dtype=float)


def predict_gp_candidate_outputs(models, x):
    proba = models.clf.predict_proba(x)
    fail_idx = list(models.clf.classes_).index(1)
    p_fail = proba[:, fail_idx]; p_pass = 1-p_fail
    boundary = np.clip(1 - 2*np.abs(p_fail-0.5), 0, 1)
    clf_unc = np.zeros_like(boundary)
    if models.has_tmax_model:
        t_pred, t_std = models.reg_tmax.predict(x, return_std=True)
    else:
        t_pred = np.zeros(x.shape[0]); t_std = np.zeros(x.shape[0])
    return {"p_fail":p_fail, "p_pass":p_pass, "boundary_score":boundary, "clf_uncertainty":clf_unc, "tmax_pred":t_pred, "tmax_std":t_std}


def predict_mlp_candidate_outputs(models, x):
    pred = predict_mlp_ensemble(models, x)
    p_fail = pred["p_fail"]; p_pass = pred["p_pass"]
    boundary = np.clip(1 - 2*np.abs(p_fail-0.5), 0, 1)
    return {"p_fail":p_fail, "p_pass":p_pass, "boundary_score":boundary, "clf_uncertainty":pred["p_fail_std"], "tmax_pred":pred["tmax_pred"], "tmax_std":pred["tmax_std"]}


def predict_candidate_outputs(models, x):
    return predict_mlp_candidate_outputs(models, x) if getattr(models, "kind", "gp") == "mlp" else predict_gp_candidate_outputs(models, x)


def compute_acquisition_scores(candidate_df, labeled_df, x_candidate_transformed, x_train_transformed, models, config):
    pred = predict_candidate_outputs(models, x_candidate_transformed)
    local = compute_local_sparsity(x_candidate_transformed, x_train_transformed)
    combo = compute_combo_priority(candidate_df, labeled_df, "discrete_combo_id", config.MIN_SAMPLES_PER_COMBO, config.MAX_SAMPLES_PER_COMBO)
    tmax_scaled = safe_minmax_scale(pred["tmax_pred"])
    tmax_unc = safe_minmax_scale(pred["tmax_std"])
    clf_unc = safe_minmax_scale(pred["clf_uncertainty"])
    pwin = p_window_score(pred["p_pass"], config.PASS_WINDOW_LOW, config.PASS_WINDOW_HIGH, config.PASS_WINDOW_CENTER)
    bw = config.BOUNDARY_WEIGHTS_MLP if getattr(models, "kind", "gp") == "mlp" else config.BOUNDARY_WEIGHTS_GP
    tw, uw = config.PASS_HIGH_TMAX_WEIGHTS, config.UNCERTAINTY_SPARSE_WEIGHTS
    acq_b = bw["boundary"]*pred["boundary_score"] + bw["clf_uncertainty"]*clf_unc + bw["local_sparsity"]*local + bw["combo_priority"]*combo
    acq_t = tw["tmax"]*tmax_scaled + tw["pass_window"]*pwin + tw["tmax_uncertainty"]*tmax_unc + tw["local_sparsity"]*local + tw["combo_priority"]*combo
    acq_u = uw["clf_uncertainty"]*clf_unc + uw["tmax_uncertainty"]*tmax_unc + uw["local_sparsity"]*local + uw["combo_priority"]*combo
    out = candidate_df.copy()
    out["selected_model_kind"] = getattr(models, "kind", "gp")
    out["p_fail"] = pred["p_fail"]; out["p_pass"] = pred["p_pass"]
    out["boundary_score"] = pred["boundary_score"]
    out["clf_uncertainty_raw"] = pred["clf_uncertainty"]
    out["clf_uncertainty_scaled"] = clf_unc
    out["tmax_pred_given_pass"] = pred["tmax_pred"]
    out["tmax_std_given_pass"] = pred["tmax_std"]
    out["tmax_scaled"] = tmax_scaled
    out["pass_window_score"] = pwin
    out["local_sparsity"] = local
    out["combo_priority"] = combo
    out["acq_boundary"] = acq_b
    out["acq_pass_high_tmax"] = acq_t
    out["acq_uncertainty_sparse"] = acq_u
    return out
