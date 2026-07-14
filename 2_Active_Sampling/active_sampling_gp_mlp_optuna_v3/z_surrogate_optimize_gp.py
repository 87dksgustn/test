import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config
from acquisition import predict_gp_outputs
from data_loader import load_labeled_data, validate_passfail_labels, validate_required_columns
from discrete_space import attach_discrete_combo_id, generate_valid_discrete_combinations
from models_gp import fit_gp_models
from optuna_tuning import maybe_tune_models
from preprocessing import build_preprocessor, make_extra_targets, make_xy

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Optuna is required for z_surrogate_optimize_gp.py") from exc


OUTPUT_DIR = Path("outputs") / "z_surrogate_optimize_gp"
RESULTS_CSV = OUTPUT_DIR / "best_by_combo.csv"
TOP_RESULTS_CSV = OUTPUT_DIR / "top_overall_feasible.csv"
TOP_PER_BIN_RESULTS_CSV = OUTPUT_DIR / "top_by_celld_bin.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

TRIALS_PER_COMBO = 80
TOP_K_OVERALL = 20
TOP_K_PER_CELLD_BIN = 10
OBJECTIVE_PTP_MAX = 0.50
TRUST_WEIGHTS = {
    "feasibility_margin": 0.35,
    "tmax_uncertainty": 0.30,
    "clf_uncertainty": 0.20,
    "data_support": 0.15,
}


def _count_from_ratio(total, labels, ratio_by_label):
    raw = {k: total * float(ratio_by_label.get(k, 0.0)) for k in labels}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = int(total) - sum(counts.values())
    order = sorted(labels, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:rem]:
        counts[k] += 1
    return counts


def _fallback_path(path):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_csv_with_fallback(df, path):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        alt = _fallback_path(path)
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        return alt


def write_json_with_fallback(obj, path):
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)
        return path
    except PermissionError:
        alt = _fallback_path(path)
        with alt.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)
        return alt


def load_training_data(cfg):
    df = load_labeled_data("initial_dataset.csv")
    validate_required_columns(
        df,
        cfg.CONTINUOUS_COLS,
        cfg.DISCRETE_COLS,
        cfg.TPNoTP_COL,
        cfg.TMAX_COL,
        cfg.OTHER_REGRESSION_COLS,
        cfg.TIME_FEATURE_COLS,
    )
    validate_passfail_labels(df, cfg.TPNoTP_COL, cfg.PASS_LABEL, cfg.FAIL_LABEL)
    valid_combos = generate_valid_discrete_combinations(cfg.DISCRETE_LEVELS, cfg.DISCRETE_COLS, cfg.S_PREFIX)
    df = attach_discrete_combo_id(df, valid_combos, cfg.DISCRETE_COLS)
    return df, valid_combos


def fit_surrogate_models(df, cfg):
    x_raw, y_class, y_tmax = make_xy(df, cfg.CONTINUOUS_COLS, cfg.DISCRETE_COLS, cfg.TPNoTP_COL, cfg.TMAX_COL)
    y_extra, _ = make_extra_targets(df, cfg.OTHER_REGRESSION_COLS, cfg.TIME_FEATURE_COLS)
    pre = build_preprocessor(cfg.CONTINUOUS_COLS, cfg.DISCRETE_COLS)
    x_train = pre.fit_transform(x_raw)
    nn_model = NearestNeighbors(n_neighbors=1).fit(x_train)
    tuned = maybe_tune_models(df, x_train, y_class, y_tmax, y_extra, cfg)
    models = fit_gp_models(
        x_train,
        y_class,
        y_tmax,
        pass_label=cfg.PASS_LABEL,
        random_state=cfg.RANDOM_SEED,
        gp_params=tuned.get("gp_params"),
        tmax_params=tuned.get("tmax_params"),
        clf_uncertainty_mode=getattr(cfg, "GP_CLF_UNCERTAINTY_MODE", "none"),
        clf_ensemble_size=getattr(cfg, "GP_CLF_ENSEMBLE_SIZE", 5),
        clf_ensemble_sample_ratio=getattr(cfg, "GP_CLF_ENSEMBLE_SAMPLE_RATIO", 0.8),
        clf_ensemble_stratified=getattr(cfg, "GP_CLF_ENSEMBLE_STRATIFIED", True),
    )
    return pre, x_train, nn_model, models, tuned


def build_celld_trial_quota(df, cfg, total_trials):
    col = getattr(cfg, "TP_RATIO_CELLD_COL", getattr(cfg, "NOTP_HIGHTMAX_CELLD_COL", "A_Cell_D"))
    bins = list(getattr(cfg, "TP_RATIO_CELLD_BINS", getattr(cfg, "NOTP_HIGHTMAX_CELLD_BINS", [])))
    labels = list(getattr(cfg, "TP_RATIO_CELLD_BIN_LABELS", getattr(cfg, "NOTP_HIGHTMAX_CELLD_BIN_LABELS", [])))
    if len(bins) < 2 or len(labels) != len(bins) - 1:
        return None

    use = df[[col, cfg.TPNoTP_COL]].dropna(subset=[col, cfg.TPNoTP_COL]).copy()
    tp = use.loc[use[cfg.TPNoTP_COL] == cfg.TP_LABEL]
    if len(tp) == 0:
        ratio = {label: 1.0 / len(labels) for label in labels}
    else:
        cat = pd.cut(tp[col], bins=bins, labels=labels, include_lowest=True, right=False)
        cnt = tp.groupby(cat, observed=False).size().reindex(labels, fill_value=0)
        total = int(cnt.sum())
        ratio = {
            label: (float(cnt.loc[label]) / total if total > 0 else 1.0 / len(labels))
            for label in labels
        }
        if sum(ratio.values()) <= 0:
            ratio = {label: 1.0 / len(labels) for label in labels}

    return {
        "col": col,
        "bins": bins,
        "labels": labels,
        "ratio_by_label": ratio,
        "trial_quota_by_label": _count_from_ratio(int(total_trials), labels, ratio),
    }


def build_top_per_bin(df, sort_cols, top_k):
    if len(df) == 0 or "cell_d_bin_label" not in df.columns:
        return pd.DataFrame()
    parts = []
    for label, part in df.groupby("cell_d_bin_label", sort=False):
        ranked = part.sort_values(sort_cols, ascending=[False, False]).head(int(top_k)).copy()
        ranked.insert(0, "cell_d_bin_rank", np.arange(1, len(ranked) + 1, dtype=int))
        parts.append(ranked)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def evaluate_point(preprocessor, nn_model, models, cfg, combo_row, cont_values):
    row = {col: float(cont_values[col]) for col in cfg.CONTINUOUS_COLS}
    for col in cfg.DISCRETE_COLS:
        row[col] = combo_row[col]
    frame = pd.DataFrame([row])
    x = preprocessor.transform(frame[cfg.CONTINUOUS_COLS + cfg.DISCRETE_COLS])
    nn_distance = float(nn_model.kneighbors(x, return_distance=True)[0][0][0])
    pred = predict_gp_outputs(models, x)
    p_tp = float(pred["p_tp"][0])
    p_notp = float(pred["p_notp"][0])
    tmax_pred = float(pred["tmax_pred"][0])
    tmax_std = float(pred["tmax_std"][0])
    clf_unc = float(pred["clf_uncertainty"][0])
    feasible = p_tp <= OBJECTIVE_PTP_MAX
    feasibility_margin = OBJECTIVE_PTP_MAX - p_tp
    score = tmax_pred if feasible else (-1e6 * (p_tp - OBJECTIVE_PTP_MAX))
    return {
        **row,
        "p_tp": p_tp,
        "p_notp": p_notp,
        "tmax_pred_given_notp": tmax_pred,
        "tmax_std_given_notp": tmax_std,
        "clf_uncertainty_raw": clf_unc,
        "nn_distance_raw": nn_distance,
        "feasibility_margin": feasibility_margin,
        "robust_tmax_lower_95": tmax_pred - 1.96 * tmax_std,
        "is_feasible": bool(feasible),
        "objective_score": float(score),
    }


def optimize_single_combo(preprocessor, nn_model, models, cfg, combo_row, n_trials, celld_trial_rule=None):
    seed_offset = int(str(combo_row["discrete_combo_id"]).split("_")[-1])
    evaluations = []

    if isinstance(celld_trial_rule, dict):
        quota_col = celld_trial_rule["col"]
        labels = list(celld_trial_rule["labels"])
        bins = list(celld_trial_rule["bins"])
        ratio_by_label = dict(celld_trial_rule["ratio_by_label"])
        quota_by_label = dict(celld_trial_rule["trial_quota_by_label"])
        label_bounds = {
            label: (float(bins[i]), float(bins[i + 1]))
            for i, label in enumerate(labels)
        }
    else:
        quota_col = None
        labels = []
        ratio_by_label = {}
        quota_by_label = {}
        label_bounds = {}

    def run_study(trials_for_bin, quota_label=None):
        if int(trials_for_bin) <= 0:
            return
        label_index = labels.index(quota_label) if quota_label in labels else 0
        sampler = optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED + seed_offset * 101 + label_index)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            cont_values = {}
            for col in cfg.CONTINUOUS_COLS:
                lo, hi = cfg.CONTINUOUS_BOUNDS[col]
                if quota_label is not None and col == quota_col:
                    lo, hi = label_bounds[quota_label]
                cont_values[col] = trial.suggest_float(col, float(lo), float(hi))
            evaluated = evaluate_point(preprocessor, nn_model, models, cfg, combo_row, cont_values)
            if quota_label is not None:
                evaluated["cell_d_bin_label"] = quota_label
                evaluated["cell_d_bin_weight"] = float(ratio_by_label.get(quota_label, 0.0))
                evaluated["cell_d_bin_trial_quota"] = int(quota_by_label.get(quota_label, 0))
            evaluations.append(evaluated)
            return evaluated["objective_score"]

        study.optimize(objective, n_trials=int(trials_for_bin), show_progress_bar=False)

    if quota_col is not None:
        for quota_label in labels:
            run_study(quota_by_label.get(quota_label, 0), quota_label)
    else:
        run_study(n_trials)

    eval_df = pd.DataFrame(evaluations)
    if len(eval_df) == 0:
        return None, eval_df

    eval_df.insert(0, "discrete_combo_id", combo_row["discrete_combo_id"])

    feasible_df = eval_df.loc[eval_df["is_feasible"]].copy()
    if len(feasible_df):
        best = feasible_df.sort_values(["tmax_pred_given_notp", "p_notp"], ascending=[False, False]).iloc[0].to_dict()
    else:
        best = eval_df.sort_values(["p_tp", "tmax_pred_given_notp"], ascending=[True, False]).iloc[0].to_dict()
    best["n_trials"] = int(n_trials)
    best["n_feasible_trials"] = int(feasible_df.shape[0])
    return best, eval_df


def _safe_unit_scale(series, higher_is_better=True):
    values = pd.Series(series, dtype=float)
    if values.empty:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        scaled = pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    else:
        scaled = (values - vmin) / (vmax - vmin)
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.clip(0.0, 1.0)


def add_trust_metrics(df):
    if len(df) == 0:
        return df
    out = df.copy()
    out["trust_feasibility_score"] = _safe_unit_scale(out["feasibility_margin"], higher_is_better=True)
    out["trust_tmax_unc_score"] = _safe_unit_scale(out["tmax_std_given_notp"], higher_is_better=False)
    out["trust_clf_unc_score"] = _safe_unit_scale(out["clf_uncertainty_raw"], higher_is_better=False)
    out["trust_support_score"] = _safe_unit_scale(out["nn_distance_raw"], higher_is_better=False)
    out["trust_score"] = (
        TRUST_WEIGHTS["feasibility_margin"] * out["trust_feasibility_score"]
        + TRUST_WEIGHTS["tmax_uncertainty"] * out["trust_tmax_unc_score"]
        + TRUST_WEIGHTS["clf_uncertainty"] * out["trust_clf_unc_score"]
        + TRUST_WEIGHTS["data_support"] * out["trust_support_score"]
    )
    out["trust_level"] = pd.cut(
        out["trust_score"],
        bins=[-np.inf, 0.45, 0.67, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    return out


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, valid_combos = load_training_data(config)
    preprocessor, x_train, nn_model, models, tuned = fit_surrogate_models(df, config)
    celld_trial_rule = build_celld_trial_quota(df, config, TRIALS_PER_COMBO)
    if celld_trial_rule is not None:
        print("[INFO] Cell_D trial ratio-by-bin:")
        print(json.dumps(celld_trial_rule["ratio_by_label"], indent=2, ensure_ascii=False))
        print("[INFO] Cell_D trial quota-by-bin:")
        print(json.dumps(celld_trial_rule["trial_quota_by_label"], indent=2, ensure_ascii=False))

    best_rows = []
    all_evals = []
    for _, combo_row in valid_combos.iterrows():
        best, eval_df = optimize_single_combo(
            preprocessor,
            nn_model,
            models,
            config,
            combo_row,
            TRIALS_PER_COMBO,
            celld_trial_rule,
        )
        if best is not None:
            best_rows.append(best)
        if len(eval_df):
            all_evals.append(eval_df)

    best_df = pd.DataFrame(best_rows)
    all_eval_df = pd.concat(all_evals, ignore_index=True) if all_evals else pd.DataFrame()
    all_eval_df = add_trust_metrics(all_eval_df)
    feasible_overall = all_eval_df.loc[all_eval_df["is_feasible"]].copy() if len(all_eval_df) else pd.DataFrame()

    sort_cols = ["tmax_pred_given_notp", "p_notp"]
    best_path = None
    if len(best_df):
        best_df = add_trust_metrics(best_df)
        best_df = best_df.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)
        best_path = write_csv_with_fallback(best_df, RESULTS_CSV)

    top_path = None
    if len(feasible_overall):
        top_df = feasible_overall.sort_values(sort_cols, ascending=[False, False]).head(TOP_K_OVERALL).reset_index(drop=True)
        top_path = write_csv_with_fallback(top_df, TOP_RESULTS_CSV)
    else:
        top_df = pd.DataFrame()

    top_per_bin_path = None
    top_per_bin_df = build_top_per_bin(feasible_overall, sort_cols, TOP_K_PER_CELLD_BIN)
    if len(top_per_bin_df):
        top_per_bin_path = write_csv_with_fallback(top_per_bin_df, TOP_PER_BIN_RESULTS_CSV)

    summary = {
        "input_csv": "initial_dataset.csv",
        "model_kind": "gp",
        "objective": {
            "constraint": f"p_tp <= {OBJECTIVE_PTP_MAX}",
            "maximize": "tmax_pred_given_notp",
        },
        "n_training_rows": int(len(df)),
        "n_valid_combos": int(len(valid_combos)),
        "trials_per_combo": int(TRIALS_PER_COMBO),
        "top_k_per_celld_bin": int(TOP_K_PER_CELLD_BIN),
        "n_total_evaluations": int(len(all_eval_df)),
        "n_total_feasible_evaluations": int(feasible_overall.shape[0]) if len(all_eval_df) else 0,
        "gp_params": tuned.get("gp_params"),
        "tmax_params": tuned.get("tmax_params"),
        "trust_weights": TRUST_WEIGHTS,
        "cell_d_trial_quota": celld_trial_rule,
        "training_nn_count": int(x_train.shape[0]),
        "runtime_sec": round(time.time() - t0, 3),
        "output_files": {
            "best_by_combo": str(best_path) if best_path else None,
            "top_overall_feasible": str(top_path) if top_path else None,
            "top_by_celld_bin": str(top_per_bin_path) if top_per_bin_path else None,
        },
    }
    if len(best_df):
        summary["best_solution"] = best_df.iloc[0].to_dict()

    summary_path = SUMMARY_JSON
    summary["output_files"]["summary"] = str(summary_path)
    summary_path = write_json_with_fallback(summary, summary_path)
    summary["output_files"]["summary"] = str(summary_path)
    if summary_path != SUMMARY_JSON:
        write_json_with_fallback(summary, summary_path)

    print("[INFO] Surrogate optimization complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(best_df):
        print("[INFO] Top 10 best-by-combo solutions")
        show_cols = [
            "discrete_combo_id",
            *config.DISCRETE_COLS,
            *config.CONTINUOUS_COLS,
            "p_tp",
            "p_notp",
            "tmax_pred_given_notp",
            "feasibility_margin",
            "tmax_std_given_notp",
            "clf_uncertainty_raw",
            "nn_distance_raw",
            "trust_score",
            "trust_level",
            "is_feasible",
        ]
        print(best_df[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()