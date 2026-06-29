from evaluation import model_score
from models_mlp import evaluate_mlp_cv, fit_mlp_ensemble, TORCH_AVAILABLE


def mlp_eligibility_report(df, passfail_col, combo_col, config):
    n_total = len(df)
    n_pass = int((df[passfail_col] == config.PASS_LABEL).sum())
    n_fail = int((df[passfail_col] == config.FAIL_LABEL).sum())
    min_combo = int(df[combo_col].value_counts().min()) if len(df) else 0
    r = {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "min_combo_count": min_combo,
        "torch_available": bool(TORCH_AVAILABLE),
        "pass_total": n_total >= config.MLP_MIN_TOTAL_SAMPLES,
        "pass_pass_count": n_pass >= config.MLP_MIN_PASS_SAMPLES,
        "pass_fail_count": n_fail >= config.MLP_MIN_FAIL_SAMPLES,
        "pass_combo_floor": min_combo >= config.MLP_MIN_SAMPLES_PER_COMBO,
        "pass_torch": bool(TORCH_AVAILABLE),
    }
    r["eligible"] = all([r["pass_total"], r["pass_pass_count"], r["pass_fail_count"], r["pass_combo_floor"], r["pass_torch"]])
    return r


def select_and_fit_model(df, x_train, y_class, y_tmax, y_extra, extra_cols, gp_models, gp_cv_result, config):
    mode = config.MODEL_MODE.lower()
    report = {
        "model_mode": mode,
        "selected_model": "gp",
        "reason": "",
        "gp_cv_result": gp_cv_result,
        "gp_score": model_score(gp_cv_result, config.MODEL_SELECTION_WEIGHTS),
        "mlp_eligibility": mlp_eligibility_report(df, config.PASSFAIL_COL, "discrete_combo_id", config),
        "mlp_cv_result": None,
        "mlp_score": None,
    }
    if mode == "gp":
        report["reason"] = "MODEL_MODE='gp'. GP was forced."
        return gp_models, report
    if not report["mlp_eligibility"]["eligible"]:
        report["reason"] = "MLP is not eligible by transition conditions. Falling back to GP."
        return gp_models, report

    mlp_cv = evaluate_mlp_cv(x_train, y_class, y_tmax, y_extra, extra_cols, config, fail_label=config.FAIL_LABEL, n_splits=config.CV_SPLITS)
    mlp_score = model_score(mlp_cv, config.MODEL_SELECTION_WEIGHTS)
    report["mlp_cv_result"] = mlp_cv
    report["mlp_score"] = mlp_score

    if mode == "mlp":
        report["selected_model"] = "mlp"
        report["reason"] = "MODEL_MODE='mlp' and MLP conditions were satisfied."
        return fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, extra_cols, config, config.RANDOM_SEED), report

    gp_score = report["gp_score"]
    if mlp_score >= gp_score + config.MLP_SELECTION_MARGIN:
        report["selected_model"] = "mlp"
        report["reason"] = f"MLP selected. mlp_score={mlp_score:.4f} >= gp_score={gp_score:.4f} + margin={config.MLP_SELECTION_MARGIN:.4f}."
        return fit_mlp_ensemble(x_train, y_class, y_tmax, y_extra, extra_cols, config, config.RANDOM_SEED), report

    report["reason"] = f"GP selected. MLP did not beat GP by margin. gp_score={gp_score:.4f}, mlp_score={mlp_score:.4f}."
    return gp_models, report
