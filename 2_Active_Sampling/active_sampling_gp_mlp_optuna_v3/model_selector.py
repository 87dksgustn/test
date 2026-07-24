import numpy as np
from sklearn.model_selection import train_test_split

from evaluation import evaluate_gpc_cv
from metrics_utils import weighted_score, classification_metrics, regression_metrics
from models_gp import fit_gp_models
from models_mlp import TORCH_AVAILABLE, evaluate_mlp_cv, fit_mlp_ensemble
from acquisition import predict_outputs

def mlp_eligibility_report(df, config):
    n_total = len(df)
    n_notp = int((df[config.PASSFAIL_COL] == config.PASS_LABEL).sum())
    n_tp = int((df[config.PASSFAIL_COL] == config.FAIL_LABEL).sum())
    min_combo_count = int(df["discrete_combo_id"].value_counts().min()) if n_total else 0
    checks = {
        "n_total": n_total, "n_notp": n_notp, "n_tp": n_tp, "min_combo_count": min_combo_count,
        "torch_available": bool(TORCH_AVAILABLE),
        "notp_total": n_total >= config.MLP_MIN_TOTAL_SAMPLES,
        "notp_count": n_notp >= config.MLP_MIN_PASS_SAMPLES,
        "tp_count": n_tp >= config.MLP_MIN_FAIL_SAMPLES,
        "combo_floor": min_combo_count >= config.MLP_MIN_SAMPLES_PER_COMBO,
        "torch_ready": bool(TORCH_AVAILABLE),
    }
    checks["eligible"] = all([checks["notp_total"], checks["notp_count"], checks["tp_count"], checks["combo_floor"], checks["torch_ready"]])
    return checks


def evaluate_holdout_model(model_kind, x_train, y_class, y_tmax, y_extra, config, gp_params=None, tmax_params=None, mlp_params=None):
    if len(y_class) < 20:
        return {"error": "Not enough samples for holdout evaluation."}

    holdout_size = float(getattr(config, "MODEL_COMPARE_HOLDOUT_TEST_SIZE", 0.20))
    holdout_size = min(0.5, max(0.1, holdout_size))
    idx = np.arange(len(y_class))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=holdout_size,
        random_state=config.RANDOM_SEED,
        stratify=y_class,
    )
    xtr, xva = x_train[tr_idx], x_train[va_idx]
    ytr, yva = y_class[tr_idx], y_class[va_idx]
    ttr, tva = y_tmax[tr_idx], y_tmax[va_idx]
    if y_extra is None:
        etr = eva = None
    else:
        etr, eva = y_extra[tr_idx], y_extra[va_idx]

    if model_kind == "gp":
        model = fit_gp_models(
            xtr,
            ytr,
            ttr,
            pass_label=config.PASS_LABEL,
            random_state=config.RANDOM_SEED,
            gp_params=gp_params,
            tmax_params=tmax_params,
            clf_uncertainty_mode=getattr(config, "GP_CLF_UNCERTAINTY_MODE", "none"),
            clf_ensemble_size=getattr(config, "GP_CLF_ENSEMBLE_SIZE", 5),
            clf_ensemble_sample_ratio=getattr(config, "GP_CLF_ENSEMBLE_SAMPLE_RATIO", 0.8),
            clf_ensemble_stratified=getattr(config, "GP_CLF_ENSEMBLE_STRATIFIED", True),
            use_ard=getattr(config, "GP_USE_ARD", True),
        )
    else:
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch is unavailable for MLP holdout."}
        model = fit_mlp_ensemble(xtr, ytr, ttr, etr, config, seed=config.RANDOM_SEED, params=mlp_params)

    pred = predict_outputs(model, xva)
    y_pred = (np.asarray(pred["p_tp"], dtype=float) >= 0.5).astype(int)
    out = classification_metrics(yva, y_pred, tp_label=config.FAIL_LABEL)

    pass_mask = (yva == config.PASS_LABEL)
    if int(pass_mask.sum()) > 0:
        out.update(regression_metrics(tva[pass_mask], np.asarray(pred["tmax_pred"], dtype=float)[pass_mask]))
    else:
        out.update({"tmax_mae": np.nan, "tmax_rmse": np.nan, "tmax_r2": np.nan})

    out["holdout_n"] = int(len(yva))
    out["tmax_eval_n"] = int(pass_mask.sum())
    out["weighted_score"] = float(weighted_score(out, config.MODEL_SELECTION_WEIGHTS))
    return out

def select_and_fit_model(df, x_train, y_class, y_tmax, y_extra, config, tuned_params):
    mode = config.MODEL_MODE.lower()
    gp_params = tuned_params.get("gp_params")
    tmax_params = tuned_params.get("tmax_params")
    mlp_params = tuned_params.get("mlp_params")
    gp_cv = evaluate_gpc_cv(x_train, y_class, y_tmax=y_tmax, pass_label=config.PASS_LABEL, tp_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=gp_params, random_state=config.RANDOM_SEED, use_ard=getattr(config, "GP_USE_ARD", True))
    gp_models = fit_gp_models(
        x_train,
        y_class,
        y_tmax,
        pass_label=config.PASS_LABEL,
        random_state=config.RANDOM_SEED,
        gp_params=gp_params,
        tmax_params=tmax_params,
        clf_uncertainty_mode=getattr(config, "GP_CLF_UNCERTAINTY_MODE", "none"),
        clf_ensemble_size=getattr(config, "GP_CLF_ENSEMBLE_SIZE", 5),
        clf_ensemble_sample_ratio=getattr(config, "GP_CLF_ENSEMBLE_SAMPLE_RATIO", 0.8),
        clf_ensemble_stratified=getattr(config, "GP_CLF_ENSEMBLE_STRATIFIED", True),
        use_ard=getattr(config, "GP_USE_ARD", True),
    )
    gp_score = gp_cv["summary"].get("stable_score", weighted_score(gp_cv["summary"], config.MODEL_SELECTION_WEIGHTS))
    report = {
        "model_mode": mode,
        "selected_model": "gp",
        "reason": "",
        "gp_cv_result": gp_cv["summary"],
        "gp_score": gp_score,
        "mlp_eligibility": mlp_eligibility_report(df, config),
        "mlp_cv_result": None,
        "mlp_score": None,
        "gp_holdout_result": None,
        "mlp_holdout_result": None,
        "composite_weights": {
            "cv": float(getattr(config, "MODEL_COMPARE_CV_WEIGHT", 0.30)),
            "holdout": float(getattr(config, "MODEL_COMPARE_HOLDOUT_WEIGHT", 0.70)),
        },
        "gp_composite_score": None,
        "mlp_composite_score": None,
        "tuned_gp_params": gp_params,
        "tuned_mlp_params": mlp_params,
        "tuned_tmax_params": tmax_params,
    }
    fold_results = {"gp": gp_cv}
    if mode == "gp":
        report["reason"] = "MODEL_MODE='gp'. GP was forced."
        return gp_models, report, fold_results
    if not report["mlp_eligibility"]["eligible"]:
        report["reason"] = "MLP is not eligible. Falling back to GP."
        return gp_models, report, fold_results
    mlp_cv = evaluate_mlp_cv(x_train, y_class, y_tmax, y_extra, config, tp_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=mlp_params)
    mlp_score = mlp_cv["summary"].get("stable_score", weighted_score(mlp_cv["summary"], config.MODEL_SELECTION_WEIGHTS))
    report["mlp_cv_result"] = mlp_cv["summary"]; report["mlp_score"] = mlp_score
    fold_results["mlp"] = mlp_cv

    gp_holdout = evaluate_holdout_model("gp", x_train, y_class, y_tmax, y_extra, config, gp_params=gp_params, tmax_params=tmax_params, mlp_params=mlp_params)
    mlp_holdout = evaluate_holdout_model("mlp", x_train, y_class, y_tmax, y_extra, config, gp_params=gp_params, tmax_params=tmax_params, mlp_params=mlp_params)
    report["gp_holdout_result"] = gp_holdout
    report["mlp_holdout_result"] = mlp_holdout

    if "error" not in gp_holdout and "error" not in mlp_holdout:
        cv_w = float(report["composite_weights"]["cv"])
        holdout_w = float(report["composite_weights"]["holdout"])
        denom = cv_w + holdout_w
        if denom <= 1e-12:
            cv_w = 0.30
            holdout_w = 0.70
            denom = 1.0
        cv_w /= denom
        holdout_w /= denom
        report["composite_weights"] = {"cv": cv_w, "holdout": holdout_w}
        report["gp_composite_score"] = float(cv_w * float(gp_score) + holdout_w * float(gp_holdout.get("weighted_score", float("-inf"))))
        report["mlp_composite_score"] = float(cv_w * float(mlp_score) + holdout_w * float(mlp_holdout.get("weighted_score", float("-inf"))))

    gp_decision_score = report.get("gp_composite_score")
    mlp_decision_score = report.get("mlp_composite_score")
    decision_basis = "cv_stable_score"
    if gp_decision_score is None or mlp_decision_score is None:
        gp_decision_score = gp_score
        mlp_decision_score = mlp_score
    else:
        decision_basis = "composite_score"

    if mode == "mlp" or float(mlp_decision_score) >= float(gp_decision_score) + config.MLP_SELECTION_MARGIN:
        mlp_bundle = fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, config, seed=config.RANDOM_SEED, params=mlp_params)
        report["selected_model"] = "mlp"
        report["reason"] = f"MLP selected by force mode or auto {decision_basis} comparison."
        return mlp_bundle, report, fold_results
    report["reason"] = (
        f"GP selected. MLP {decision_basis} did not exceed GP by margin. "
        f"gp={float(gp_decision_score):.4f}, mlp={float(mlp_decision_score):.4f}."
    )
    return gp_models, report, fold_results
