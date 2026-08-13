import numpy as np
from sklearn.model_selection import train_test_split

from evaluation import evaluate_gpc_cv, evaluate_gp_cv_with_extra
from metrics_utils import weighted_score, classification_metrics, regression_metrics
from models_gp import fit_gp_models, HybridModels
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


def evaluate_holdout_model(model_kind, x_train, y_class, y_tmax, y_extra, config, gp_params=None, tmax_params=None, mlp_params=None, extra_cols=None):
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
        # Use isotropic kernel for holdout evaluation (faster), ARD only for final fit
        use_ard_for_holdout = getattr(config, "GP_MODEL_SELECTION_USE_ARD", False)
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
            use_ard=use_ard_for_holdout,
            y_extra=etr,
            extra_cols=extra_cols,
        )
    else:
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch is unavailable for MLP holdout."}
        model = fit_mlp_ensemble(xtr, ytr, ttr, etr, config, seed=config.RANDOM_SEED, params=mlp_params, extra_cols=extra_cols)

    pred = predict_outputs(model, xva)
    y_pred = (np.asarray(pred["p_tp"], dtype=float) >= 0.5).astype(int)
    out = classification_metrics(yva, y_pred, tp_label=config.FAIL_LABEL)

    pass_mask = (yva == config.PASS_LABEL)
    if int(pass_mask.sum()) > 0:
        out.update(regression_metrics(tva[pass_mask], np.asarray(pred["tmax_pred"], dtype=float)[pass_mask]))
    else:
        out.update({"tmax_mae": np.nan, "tmax_rmse": np.nan, "tmax_r2": np.nan})

    # Extra outputs metrics (evaluation-only, not used for scoring or model selection)
    out["extra_metrics"] = {}
    if eva is not None and extra_cols is not None and int(pass_mask.sum()) > 0:
        extra_pred = pred.get("extra_pred")  # MLP: from extra head, GP: separate call
        if extra_pred is None and hasattr(model, "reg_extra") and model.reg_extra:
            # GP model: predict extra outputs from separate regressors
            extra_pred_list = []
            for col in extra_cols:
                if col in model.reg_extra:
                    ep = model.reg_extra[col].predict(xva[pass_mask])
                    extra_pred_list.append(ep)
            if extra_pred_list:
                extra_pred = np.column_stack(extra_pred_list)  # (n_notp, n_extra)
        if extra_pred is not None:
            extra_pred_notp = extra_pred[pass_mask] if len(extra_pred) == len(yva) else extra_pred
            for i, col in enumerate(extra_cols):
                try:
                    col_metrics = regression_metrics(eva[pass_mask, i], extra_pred_notp[:, i])
                    out["extra_metrics"][col] = {
                        "rmse": float(col_metrics.get("tmax_rmse", np.nan)),
                        "r2": float(col_metrics.get("tmax_r2", np.nan)),
                        "mae": float(col_metrics.get("tmax_mae", np.nan)),
                    }
                except Exception:
                    out["extra_metrics"][col] = {"rmse": np.nan, "r2": np.nan, "mae": np.nan}

    out["holdout_n"] = int(len(yva))
    out["tmax_eval_n"] = int(pass_mask.sum())
    out["weighted_score"] = float(weighted_score(out, config.MODEL_SELECTION_WEIGHTS))
    return out

def select_and_fit_model(df, x_train, y_class, y_tmax, y_extra, config, tuned_params, extra_cols=None):
    mode = config.MODEL_MODE.lower()
    
    # Hybrid mode: delegate to select_hybrid_model
    if mode == "hybrid":
        return select_hybrid_model(df, x_train, y_class, y_tmax, y_extra, config, tuned_params, extra_cols)
    
    gp_params = tuned_params.get("gp_params")
    tmax_params = tuned_params.get("tmax_params")
    mlp_params = tuned_params.get("mlp_params")
    extra_cols = extra_cols or []
    # Use isotropic kernel for CV evaluation (faster), ARD only for final fit
    use_ard_for_cv = getattr(config, "GP_MODEL_SELECTION_USE_ARD", False)
    gp_cv = evaluate_gpc_cv(x_train, y_class, y_tmax=y_tmax, pass_label=config.PASS_LABEL, tp_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=gp_params, random_state=config.RANDOM_SEED, use_ard=use_ard_for_cv)
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
        y_extra=y_extra,
        extra_cols=extra_cols,
    )
    gp_score = gp_cv["summary"].get("stable_score", weighted_score(gp_cv["summary"], config.MODEL_SELECTION_WEIGHTS))
    report = {
        "model_mode": mode,
        "selected_model": "gp",
        "reason": "",
        "evaluation_method": "cv",
        "cv_splits": gp_cv["summary"].get("cv_splits", config.CV_SPLITS),
        "gp_cv_result": gp_cv["summary"],
        "gp_score": gp_score,
        "mlp_eligibility": mlp_eligibility_report(df, config),
        "mlp_cv_result": None,
        "mlp_score": None,
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
    mlp_cv = evaluate_mlp_cv(x_train, y_class, y_tmax, y_extra, config, tp_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS, weights=config.MODEL_SELECTION_WEIGHTS, std_penalty=config.CV_STD_PENALTY, params=mlp_params, extra_cols=extra_cols)
    mlp_score = mlp_cv["summary"].get("stable_score", weighted_score(mlp_cv["summary"], config.MODEL_SELECTION_WEIGHTS))
    report["mlp_cv_result"] = mlp_cv["summary"]
    report["mlp_score"] = mlp_score
    fold_results["mlp"] = mlp_cv

    # Model selection based on CV stable score (100% CV, no holdout)
    gp_decision_score = gp_score
    mlp_decision_score = mlp_score
    decision_basis = "cv_stable_score"

    if mode == "mlp" or (float(mlp_decision_score) >= float(gp_decision_score) + config.MLP_SELECTION_MARGIN):
        mlp_bundle = fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, config, seed=config.RANDOM_SEED, params=mlp_params, extra_cols=extra_cols)
        report["selected_model"] = "mlp"
        report["reason"] = f"MLP selected by force mode or auto {decision_basis} comparison. GP={gp_decision_score:.4f}, MLP={mlp_decision_score:.4f}."
        return mlp_bundle, report, fold_results
    
    report["reason"] = (
            f"GP selected. MLP {decision_basis} did not exceed GP by margin. "
            f"gp={float(gp_decision_score):.4f}, mlp={float(mlp_decision_score):.4f}."
        )
    return gp_models, report, fold_results


def select_hybrid_model(df, x_train, y_class, y_tmax, y_extra, config, tuned_params, extra_cols=None):
    """
    Hybrid model selection: select best model independently for each task using CV.
    
    Tasks:
      - classifier: selected by CV classification metrics (weighted tp_recall + tp_f1)
      - tmax: selected by CV regression metric (r2)
      - each extra output: selected independently by CV rmse
    
    Returns:
      - HybridModels bundle with per-task model selection
      - report with selection details
      - fold_results for diagnostics
    """
    gp_params = tuned_params.get("gp_params")
    tmax_params = tuned_params.get("tmax_params")
    mlp_params = tuned_params.get("mlp_params")
    extra_cols = extra_cols or []
    
    # Check MLP eligibility
    mlp_elig = mlp_eligibility_report(df, config)
    if not mlp_elig["eligible"]:
        # Fallback to GP for all tasks if MLP is not eligible
        gp_models = fit_gp_models(
            x_train, y_class, y_tmax,
            pass_label=config.PASS_LABEL,
            random_state=config.RANDOM_SEED,
            gp_params=gp_params,
            tmax_params=tmax_params,
            clf_uncertainty_mode=getattr(config, "GP_CLF_UNCERTAINTY_MODE", "none"),
            clf_ensemble_size=getattr(config, "GP_CLF_ENSEMBLE_SIZE", 5),
            clf_ensemble_sample_ratio=getattr(config, "GP_CLF_ENSEMBLE_SAMPLE_RATIO", 0.8),
            clf_ensemble_stratified=getattr(config, "GP_CLF_ENSEMBLE_STRATIFIED", True),
            use_ard=getattr(config, "GP_USE_ARD", True),
            y_extra=y_extra,
            extra_cols=extra_cols,
        )
        report = {
            "model_mode": "hybrid",
            "selected_model": "gp (hybrid fallback)",
            "reason": "MLP not eligible. Fallback to GP for all tasks.",
            "mlp_eligibility": mlp_elig,
            "hybrid_selection": None,
        }
        return gp_models, report, {}
    
    # Evaluate both models with CV (including extra outputs)
    use_ard_for_cv = getattr(config, "GP_MODEL_SELECTION_USE_ARD", False)
    
    gp_cv = evaluate_gp_cv_with_extra(
        x_train, y_class, y_tmax, y_extra,
        pass_label=config.PASS_LABEL,
        tp_label=config.FAIL_LABEL,
        n_splits=config.CV_SPLITS,
        weights=config.MODEL_SELECTION_WEIGHTS,
        std_penalty=config.CV_STD_PENALTY,
        params=gp_params,
        tmax_params=tmax_params,
        random_state=config.RANDOM_SEED,
        use_ard=use_ard_for_cv,
        extra_cols=extra_cols,
    )
    
    mlp_cv = evaluate_mlp_cv(
        x_train, y_class, y_tmax, y_extra, config,
        tp_label=config.FAIL_LABEL,
        n_splits=config.CV_SPLITS,
        weights=config.MODEL_SELECTION_WEIGHTS,
        std_penalty=config.CV_STD_PENALTY,
        params=mlp_params,
        extra_cols=extra_cols,
    )
    
    if "error" in gp_cv.get("summary", {}) or "error" in mlp_cv.get("summary", {}):
        # Fallback to GP if CV failed
        gp_models = fit_gp_models(
            x_train, y_class, y_tmax,
            pass_label=config.PASS_LABEL,
            random_state=config.RANDOM_SEED,
            gp_params=gp_params,
            tmax_params=tmax_params,
            clf_uncertainty_mode=getattr(config, "GP_CLF_UNCERTAINTY_MODE", "none"),
            clf_ensemble_size=getattr(config, "GP_CLF_ENSEMBLE_SIZE", 5),
            clf_ensemble_sample_ratio=getattr(config, "GP_CLF_ENSEMBLE_SAMPLE_RATIO", 0.8),
            clf_ensemble_stratified=getattr(config, "GP_CLF_ENSEMBLE_STRATIFIED", True),
            use_ard=getattr(config, "GP_USE_ARD", True),
            y_extra=y_extra,
            extra_cols=extra_cols,
        )
        report = {
            "model_mode": "hybrid",
            "selected_model": "gp (CV error fallback)",
            "reason": f"CV evaluation failed. GP error: {gp_cv.get('summary', {}).get('error', 'none')}, MLP error: {mlp_cv.get('summary', {}).get('error', 'none')}",
            "mlp_eligibility": mlp_elig,
            "hybrid_selection": None,
        }
        return gp_models, report, {}
    
    # Get CV summaries
    gp_summary = gp_cv["summary"]
    mlp_summary = mlp_cv["summary"]
    gp_extra_summary = gp_cv.get("extra_summary", {})
    mlp_extra_summary = mlp_cv.get("extra_summary", {})
    
    # Get selection criteria from config
    clf_weights = getattr(config, "HYBRID_CLASSIFIER_WEIGHTS", {"tp_recall": 0.70, "tp_f1": 0.30})
    tmax_metric = getattr(config, "HYBRID_TMAX_METRIC", "r2")
    extra_metric = getattr(config, "HYBRID_EXTRA_METRIC", "rmse")
    clf_margin = getattr(config, "HYBRID_CLASSIFIER_MARGIN", 0.01)
    reg_margin = getattr(config, "HYBRID_REGRESSION_MARGIN", 0.02)
    
    # Helper: compute weighted classifier score from CV summary
    def compute_clf_weighted_score(cv_summary, weights):
        """Compute weighted sum of classifier metrics from CV mean values."""
        total_weight = sum(weights.values())
        if total_weight <= 0:
            total_weight = 1.0
        score = 0.0
        for metric, weight in weights.items():
            # CV summary stores mean values
            val = cv_summary.get(f"{metric}_mean", cv_summary.get(metric, 0.0))
            val = float(val) if np.isfinite(val) else 0.0
            score += (weight / total_weight) * val
        return score
    
    # Helper: compare metric values (higher_is_better determines direction)
    def compare_metric(gp_val, mlp_val, higher_is_better, margin):
        """Return 'mlp' if MLP wins by margin, else 'gp'."""
        gp_val = float(gp_val) if np.isfinite(gp_val) else float("-inf") if higher_is_better else float("inf")
        mlp_val = float(mlp_val) if np.isfinite(mlp_val) else float("-inf") if higher_is_better else float("inf")
        
        if higher_is_better:
            # MLP needs to be higher than GP + margin
            if mlp_val >= gp_val + margin:
                return "mlp"
        else:
            # MLP needs to be lower than GP * (1 - margin) for relative improvement
            if gp_val > 0 and mlp_val <= gp_val * (1 - margin):
                return "mlp"
            elif gp_val <= 0 and mlp_val < gp_val:
                return "mlp"
        return "gp"
    
    # 1. Classifier selection (weighted sum of tp_recall + tp_f1 from CV)
    gp_clf_val = compute_clf_weighted_score(gp_summary, clf_weights)
    mlp_clf_val = compute_clf_weighted_score(mlp_summary, clf_weights)
    clf_winner = compare_metric(gp_clf_val, mlp_clf_val, True, clf_margin)  # higher is better
    
    # 2. Tmax regressor selection (from CV)
    tmax_key = f"tmax_{tmax_metric}_mean" if not tmax_metric.startswith("tmax_") else f"{tmax_metric}_mean"
    tmax_key_fallback = f"tmax_{tmax_metric}" if not tmax_metric.startswith("tmax_") else tmax_metric
    gp_tmax_val = gp_summary.get(tmax_key, gp_summary.get(tmax_key_fallback, float("-inf") if tmax_metric == "r2" else float("inf")))
    mlp_tmax_val = mlp_summary.get(tmax_key, mlp_summary.get(tmax_key_fallback, float("-inf") if tmax_metric == "r2" else float("inf")))
    tmax_higher_better = tmax_metric in ["r2", "tmax_r2"]
    tmax_winner = compare_metric(gp_tmax_val, mlp_tmax_val, tmax_higher_better, reg_margin)
    
    # 3. Extra outputs selection (independent per column, using CV summary)
    extra_winners = {}
    extra_higher_better = extra_metric in ["r2"]
    metric_key = f"{extra_metric}_mean"
    
    for col in extra_cols:
        gp_col = gp_extra_summary.get(col, {})
        mlp_col = mlp_extra_summary.get(col, {})
        gp_val = gp_col.get(metric_key, float("inf") if not extra_higher_better else float("-inf"))
        mlp_val = mlp_col.get(metric_key, float("inf") if not extra_higher_better else float("-inf"))
        extra_winners[col] = compare_metric(gp_val, mlp_val, extra_higher_better, reg_margin)
    
    # Build hybrid selection report
    hybrid_selection = {
        "evaluation_method": "cv",
        "cv_splits": gp_summary.get("cv_splits", config.CV_SPLITS),
        "classifier": {
            "winner": clf_winner,
            "metric": "weighted(tp_recall+tp_f1)",
            "weights": clf_weights,
            "gp_value": float(gp_clf_val),
            "mlp_value": float(mlp_clf_val),
            "gp_tp_recall_mean": float(gp_summary.get("tp_recall_mean", gp_summary.get("tp_recall", 0))),
            "gp_tp_f1_mean": float(gp_summary.get("tp_f1_mean", gp_summary.get("tp_f1", 0))),
            "mlp_tp_recall_mean": float(mlp_summary.get("tp_recall_mean", mlp_summary.get("tp_recall", 0))),
            "mlp_tp_f1_mean": float(mlp_summary.get("tp_f1_mean", mlp_summary.get("tp_f1", 0))),
        },
        "tmax": {
            "winner": tmax_winner,
            "metric": tmax_metric,
            "gp_value": float(gp_tmax_val),
            "mlp_value": float(mlp_tmax_val),
        },
        "extras": {},
    }
    for col in extra_cols:
        gp_col = gp_extra_summary.get(col, {})
        mlp_col = mlp_extra_summary.get(col, {})
        hybrid_selection["extras"][col] = {
            "winner": extra_winners.get(col, "gp"),
            "metric": extra_metric,
            "gp_value": float(gp_col.get(metric_key, np.nan)),
            "mlp_value": float(mlp_col.get(metric_key, np.nan)),
            "gp_std": float(gp_col.get(f"{extra_metric}_std", np.nan)),
            "mlp_std": float(mlp_col.get(f"{extra_metric}_std", np.nan)),
        }
    
    # Fit GP model (always needed for GP components)
    gp_models = fit_gp_models(
        x_train, y_class, y_tmax,
        pass_label=config.PASS_LABEL,
        random_state=config.RANDOM_SEED,
        gp_params=gp_params,
        tmax_params=tmax_params,
        clf_uncertainty_mode=getattr(config, "GP_CLF_UNCERTAINTY_MODE", "none"),
        clf_ensemble_size=getattr(config, "GP_CLF_ENSEMBLE_SIZE", 5),
        clf_ensemble_sample_ratio=getattr(config, "GP_CLF_ENSEMBLE_SAMPLE_RATIO", 0.8),
        clf_ensemble_stratified=getattr(config, "GP_CLF_ENSEMBLE_STRATIFIED", True),
        use_ard=getattr(config, "GP_USE_ARD", True),
        y_extra=y_extra,
        extra_cols=extra_cols,
    )
    
    # Fit MLP model if any task uses MLP
    mlp_needed = clf_winner == "mlp" or tmax_winner == "mlp" or "mlp" in extra_winners.values()
    mlp_models = None
    if mlp_needed:
        mlp_models = fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, config, 
                                       seed=config.RANDOM_SEED, params=mlp_params, extra_cols=extra_cols)
    
    # Build HybridModels bundle
    clf_model = mlp_models if clf_winner == "mlp" else gp_models
    tmax_model = mlp_models if tmax_winner == "mlp" else gp_models
    
    extra_models = {}
    for col in extra_cols:
        winner = extra_winners.get(col, "gp")
        if winner == "mlp" and mlp_models is not None:
            extra_models[col] = (mlp_models, "mlp")
        else:
            extra_models[col] = (gp_models, "gp")
    
    hybrid_bundle = HybridModels(
        clf_model=clf_model,
        clf_kind=clf_winner,
        tmax_model=tmax_model,
        tmax_kind=tmax_winner,
        extra_models=extra_models,
        extra_cols=extra_cols,
        selection_report=hybrid_selection,
    )
    
    # Summary report
    report = {
        "model_mode": "hybrid",
        "selected_model": "hybrid",
        "reason": "Hybrid mode: per-task model selection based on CV.",
        "mlp_eligibility": mlp_elig,
        "gp_cv_result": gp_cv["summary"],
        "mlp_cv_result": mlp_cv["summary"],
        "gp_extra_cv_summary": gp_extra_summary,
        "mlp_extra_cv_summary": mlp_extra_summary,
        "hybrid_selection": hybrid_selection,
        "tuned_gp_params": gp_params,
        "tuned_mlp_params": mlp_params,
        "tuned_tmax_params": tmax_params,
    }
    
    fold_results = {"gp_cv": gp_cv, "mlp_cv": mlp_cv}
    return hybrid_bundle, report, fold_results
