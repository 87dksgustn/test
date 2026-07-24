import json
import math
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
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
OUTPUT_ROOT = Path("outputs") / "z_surrogate_optimize_gp"
PARETO_CSV_NAME = "pareto_all_bins.csv"
LEXICOGRAPHIC_CSV_NAME = "lexicographic_top_by_bin.csv"
PARETO_TOP_CSV_NAME = "pareto_top_by_bin.csv"
SUMMARY_JSON_NAME = "summary.json"
PARETO_PLOT_NAME = "pareto_front_4x2.png"

TRIALS_PER_COMBO = 80
TOP_K_PER_CELLD_BIN = 10
OBJECTIVE_PTP_MAX = 0.50
NSGAII_POPULATION_SIZE = 32
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


def create_opt_dir(output_root):
    pattern = re.compile(r"^Opt_(\d+)$")
    existing = []
    for p in Path(output_root).iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                existing.append(int(m.group(1)))
    next_n = 1 if not existing else max(existing) + 1
    opt_dir = Path(output_root) / f"Opt_{next_n}"
    opt_dir.mkdir(parents=True, exist_ok=False)
    return opt_dir


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
        use_ard=getattr(cfg, "GP_USE_ARD", True),
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


def build_lexicographic_top_per_bin(df, top_k):
    if len(df) == 0 or "cell_d_bin_label" not in df.columns:
        return pd.DataFrame()
    feasible = df.loc[df["is_feasible"]].copy()
    if len(feasible) == 0:
        return pd.DataFrame()
    parts = []
    sort_cols = ["p_tp", "tmax_pred_given_notp", "C_Barrier_Thx"]
    ascending = [False, False, True]
    for label, part in feasible.groupby("cell_d_bin_label", sort=False):
        ranked = part.sort_values(sort_cols, ascending=ascending).head(int(top_k)).copy()
        ranked.insert(0, "lexicographic_rank", np.arange(1, len(ranked) + 1, dtype=int))
        parts.append(ranked)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def is_pareto_efficient(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(values.shape[0], dtype=bool)
    for idx in range(values.shape[0]):
        if not keep[idx]:
            continue
        dominates = np.all(values >= values[idx], axis=1) & np.any(values > values[idx], axis=1)
        if np.any(dominates):
            keep[idx] = False
    return keep


def build_pareto_front(df):
    if len(df) == 0:
        return pd.DataFrame()
    feasible = df.loc[df["is_feasible"]].copy()
    if len(feasible) == 0:
        return pd.DataFrame()
    parts = []
    for label, part in feasible.groupby("cell_d_bin_label", sort=False):
        objective_values = np.column_stack([
            part["p_tp"].to_numpy(dtype=float),
            part["tmax_pred_given_notp"].to_numpy(dtype=float),
            -part["C_Barrier_Thx"].to_numpy(dtype=float),
        ])
        mask = is_pareto_efficient(objective_values)
        front = part.loc[mask].copy()
        front = front.sort_values(["p_tp", "tmax_pred_given_notp", "C_Barrier_Thx"], ascending=[False, False, True]).reset_index(drop=True)
        front.insert(0, "pareto_rank_in_bin", np.arange(1, len(front) + 1, dtype=int))
        parts.append(front)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_pareto_top_per_bin(df, top_k):
    if len(df) == 0:
        return pd.DataFrame()
    parts = []
    for _, part in df.groupby("cell_d_bin_label", sort=False):
        ranked = part.head(int(top_k)).copy()
        parts.append(ranked)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def save_pareto_front_figure(all_df, pareto_df, output_png, bin_labels):
    if len(all_df) == 0 or "cell_d_bin_label" not in all_df.columns:
        return None

    fig, axes = plt.subplots(len(bin_labels), 2, figsize=(16, 20), dpi=170)
    if len(bin_labels) == 1:
        axes = np.array([axes])

    for row_idx, label in enumerate(bin_labels):
        part = all_df.loc[all_df["cell_d_bin_label"] == label].copy()
        feasible = part.loc[part["is_feasible"]].copy()
        front = pareto_df.loc[pareto_df["cell_d_bin_label"] == label].copy() if len(pareto_df) else pd.DataFrame()

        ax0 = axes[row_idx, 0]
        ax1 = axes[row_idx, 1]

        ax0.scatter(feasible["C_Barrier_Thx"], feasible["tmax_pred_given_notp"], s=18, alpha=0.35, color="#8aa1b1")
        if len(front):
            ax0.scatter(front["C_Barrier_Thx"], front["tmax_pred_given_notp"], s=32, color="#c94f3d")
        ax0.set_ylabel(f"{label}\nPredicted Tmax")
        if row_idx == len(bin_labels) - 1:
            ax0.set_xlabel("C_Barrier_Thx (minimize)")
        ax0.set_title("Barrier vs Tmax")

        ax1.scatter(feasible["p_tp"], feasible["tmax_pred_given_notp"], s=18, alpha=0.35, color="#8aa1b1")
        if len(front):
            ax1.scatter(front["p_tp"], front["tmax_pred_given_notp"], s=32, color="#c94f3d")
        ax1.axvline(OBJECTIVE_PTP_MAX, color="#222222", linestyle="--", linewidth=1.0)
        if row_idx == len(bin_labels) - 1:
            ax1.set_xlabel("p_tp (maximize, < 0.5)")
        ax1.set_title("Boundary Nearness vs Tmax")

    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_out, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Cell_D Bin-wise Pareto Fronts", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_png)


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
        "constraint_violation": float(max(0.0, p_tp - OBJECTIVE_PTP_MAX)),
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
        sampler = optuna.samplers.NSGAIISampler(
            population_size=min(NSGAII_POPULATION_SIZE, max(8, int(trials_for_bin))),
            seed=cfg.RANDOM_SEED + seed_offset * 101 + label_index,
            constraints_func=lambda frozen_trial: frozen_trial.user_attrs.get("constraints", (0.0,)),
        )
        study = optuna.create_study(directions=["maximize", "maximize", "minimize"], sampler=sampler)

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
            trial.set_user_attr("constraints", (float(evaluated["constraint_violation"]),))
            evaluations.append(evaluated)
            return (
                float(evaluated["p_tp"]),
                float(evaluated["tmax_pred_given_notp"]),
                float(evaluated["C_Barrier_Thx"]),
            )

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
    eval_df["optimization_method"] = "nsga2"
    return eval_df


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


def add_boundary_priority_metrics(df):
    if len(df) == 0:
        return df
    out = df.copy()
    out["boundary_gap"] = (OBJECTIVE_PTP_MAX - out["p_tp"]).clip(lower=0.0)
    out["boundary_near_score"] = 1.0 - _safe_unit_scale(out["boundary_gap"], higher_is_better=True)
    return out


def main():
    t0 = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = create_opt_dir(OUTPUT_ROOT)
    pareto_csv_path = run_dir / PARETO_CSV_NAME
    lexicographic_csv_path = run_dir / LEXICOGRAPHIC_CSV_NAME
    pareto_top_csv_path = run_dir / PARETO_TOP_CSV_NAME
    summary_json_path = run_dir / SUMMARY_JSON_NAME
    pareto_plot_path = run_dir / PARETO_PLOT_NAME

    df, valid_combos = load_training_data(config)
    preprocessor, x_train, nn_model, models, tuned = fit_surrogate_models(df, config)
    celld_trial_rule = build_celld_trial_quota(df, config, TRIALS_PER_COMBO)
    if celld_trial_rule is not None:
        print("[INFO] Cell_D trial ratio-by-bin:")
        print(json.dumps(celld_trial_rule["ratio_by_label"], indent=2, ensure_ascii=False))
        print("[INFO] Cell_D trial quota-by-bin:")
        print(json.dumps(celld_trial_rule["trial_quota_by_label"], indent=2, ensure_ascii=False))

    all_evals = []
    for _, combo_row in valid_combos.iterrows():
        eval_df = optimize_single_combo(
            preprocessor,
            nn_model,
            models,
            config,
            combo_row,
            TRIALS_PER_COMBO,
            celld_trial_rule,
        )
        if len(eval_df):
            all_evals.append(eval_df)

    all_eval_df = pd.concat(all_evals, ignore_index=True) if all_evals else pd.DataFrame()
    all_eval_df = add_trust_metrics(all_eval_df)
    all_eval_df = add_boundary_priority_metrics(all_eval_df)
    feasible_overall = all_eval_df.loc[all_eval_df["is_feasible"]].copy() if len(all_eval_df) else pd.DataFrame()

    pareto_df = build_pareto_front(all_eval_df)
    pareto_df = add_trust_metrics(pareto_df)
    pareto_df = add_boundary_priority_metrics(pareto_df)
    pareto_top_df = build_pareto_top_per_bin(pareto_df, TOP_K_PER_CELLD_BIN)
    lexicographic_df = build_lexicographic_top_per_bin(all_eval_df, TOP_K_PER_CELLD_BIN)
    lexicographic_df = add_trust_metrics(lexicographic_df)
    lexicographic_df = add_boundary_priority_metrics(lexicographic_df)
    plot_path = save_pareto_front_figure(all_eval_df, pareto_df, pareto_plot_path, celld_trial_rule["labels"] if celld_trial_rule else sorted(all_eval_df["cell_d_bin_label"].dropna().unique().tolist()))

    pareto_path = write_csv_with_fallback(pareto_df, pareto_csv_path) if len(pareto_df) else None
    pareto_top_path = write_csv_with_fallback(pareto_top_df, pareto_top_csv_path) if len(pareto_top_df) else None
    lexicographic_path = write_csv_with_fallback(lexicographic_df, lexicographic_csv_path) if len(lexicographic_df) else None

    summary = {
        "input_csv": "initial_dataset.csv",
        "model_kind": "gp",
        "objective": {
            "constraint": f"p_tp <= {OBJECTIVE_PTP_MAX}",
            "maximize": ["p_tp", "tmax_pred_given_notp"],
            "minimize": ["C_Barrier_Thx"],
        },
        "optimizer": "NSGA-II",
        "run_dir": str(run_dir),
        "n_training_rows": int(len(df)),
        "n_valid_combos": int(len(valid_combos)),
        "trials_per_combo": int(TRIALS_PER_COMBO),
        "top_k_per_celld_bin": int(TOP_K_PER_CELLD_BIN),
        "n_total_evaluations": int(len(all_eval_df)),
        "n_total_feasible_evaluations": int(feasible_overall.shape[0]) if len(all_eval_df) else 0,
        "n_pareto_points": int(len(pareto_df)),
        "gp_params": tuned.get("gp_params"),
        "tmax_params": tuned.get("tmax_params"),
        "trust_weights": TRUST_WEIGHTS,
        "cell_d_trial_quota": celld_trial_rule,
        "training_nn_count": int(x_train.shape[0]),
        "runtime_sec": round(time.time() - t0, 3),
        "output_files": {
            "pareto_all_bins": str(pareto_path) if pareto_path else None,
            "pareto_top_by_bin": str(pareto_top_path) if pareto_top_path else None,
            "lexicographic_top_by_bin": str(lexicographic_path) if lexicographic_path else None,
            "pareto_front_figure": str(plot_path) if plot_path else None,
        },
    }
    if len(lexicographic_df):
        summary["recommended_solution"] = lexicographic_df.iloc[0].to_dict()

    summary["output_files"]["summary"] = str(summary_json_path)
    summary_path = write_json_with_fallback(summary, summary_json_path)
    summary["output_files"]["summary"] = str(summary_path)
    if summary_path != summary_json_path:
        write_json_with_fallback(summary, summary_path)

    print("[INFO] Surrogate optimization complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(lexicographic_df):
        print("[INFO] Top lexicographic solutions by Cell_D bin")
        show_cols = [
            "cell_d_bin_label",
            "lexicographic_rank",
            "discrete_combo_id",
            *config.DISCRETE_COLS,
            *config.CONTINUOUS_COLS,
            "p_tp",
            "tmax_pred_given_notp",
            "boundary_gap",
            "trust_score",
            "trust_level",
            "is_feasible",
        ]
        print(lexicographic_df[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()