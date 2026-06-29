from evaluation import evaluate_gpc_cv
from metrics_utils import weighted_score
from models_gp import fit_gp_models
from models_mlp import TORCH_AVAILABLE, evaluate_mlp_cv, fit_mlp_ensemble

def mlp_eligibility_report(df, config):
    n_total = len(df)
    n_pass = int((df[config.PASSFAIL_COL] == config.PASS_LABEL).sum())
    n_fail = int((df[config.PASSFAIL_COL] == config.FAIL_LABEL).sum())
    min_combo_count = int(df["discrete_combo_id"].value_counts().min()) if n_total else 0
    checks = {
        "n_total": n_total, "n_pass": n_pass, "n_fail": n_fail, "min_combo_count": min_combo_count,
        "torch_available": bool(TORCH_AVAILABLE),
        "pass_total": n_total >= config.MLP_MIN_TOTAL_SAMPLES,
        "pass_pass_count": n_pass >= config.MLP_MIN_PASS_SAMPLES,
        "pass_fail_count": n_fail >= config.MLP_MIN_FAIL_SAMPLES,
        "pass_combo_floor": min_combo_count >= config.MLP_MIN_SAMPLES_PER_COMBO,
        "pass_torch": bool(TORCH_AVAILABLE),
    }
    checks["eligible"] = all([checks["pass_total"], checks["pass_pass_count"], checks["pass_fail_count"], checks["pass_combo_floor"], checks["pass_torch"]])
    return checks

def select_and_fit_model(df, x_train, y_class, y_tmax, y_extra, config, tuned_params):
    mode = config.MODEL_MODE.lower()
    gp_params = tuned_params.get("gp_params")
    tmax_params = tuned_params.get("tmax_params")
    mlp_params = tuned_params.get("mlp_params")
    gp_cv = evaluate_gpc_cv(x_train, y_class, fail_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=gp_params, random_state=config.RANDOM_SEED)
    gp_models = fit_gp_models(x_train, y_class, y_tmax, pass_label=config.PASS_LABEL, random_state=config.RANDOM_SEED, gp_params=gp_params, tmax_params=tmax_params)
    gp_score = gp_cv["summary"].get("stable_score", weighted_score(gp_cv["summary"], config.MODEL_SELECTION_WEIGHTS))
    report = {"model_mode": mode, "selected_model": "gp", "reason": "", "gp_cv_result": gp_cv["summary"], "gp_score": gp_score, "mlp_eligibility": mlp_eligibility_report(df, config), "mlp_cv_result": None, "mlp_score": None, "tuned_gp_params": gp_params, "tuned_mlp_params": mlp_params, "tuned_tmax_params": tmax_params}
    fold_results = {"gp": gp_cv}
    if mode == "gp":
        report["reason"] = "MODEL_MODE='gp'. GP was forced."
        return gp_models, report, fold_results
    if not report["mlp_eligibility"]["eligible"]:
        report["reason"] = "MLP is not eligible. Falling back to GP."
        return gp_models, report, fold_results
    mlp_cv = evaluate_mlp_cv(x_train, y_class, y_tmax, y_extra, config, fail_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=mlp_params)
    mlp_score = mlp_cv["summary"].get("stable_score", weighted_score(mlp_cv["summary"], config.MODEL_SELECTION_WEIGHTS))
    report["mlp_cv_result"] = mlp_cv["summary"]; report["mlp_score"] = mlp_score
    fold_results["mlp"] = mlp_cv
    if mode == "mlp" or mlp_score >= gp_score + config.MLP_SELECTION_MARGIN:
        mlp_bundle = fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, config, seed=config.RANDOM_SEED, params=mlp_params)
        report["selected_model"] = "mlp"
        report["reason"] = "MLP selected by force mode or auto stable-score comparison."
        return mlp_bundle, report, fold_results
    report["reason"] = f"GP selected. MLP stable score did not exceed GP by margin. gp={gp_score:.4f}, mlp={mlp_score:.4f}."
    return gp_models, report, fold_results
