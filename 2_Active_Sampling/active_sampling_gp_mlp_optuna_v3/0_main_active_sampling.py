import json
import itertools
import logging
import math
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_loader import load_labeled_data, validate_required_columns, validate_passfail_labels
from discrete_space import generate_valid_discrete_combinations, attach_discrete_combo_id
from candidate_generator import generate_candidate_pool
from preprocessing import build_preprocessor, make_xy, make_extra_targets
from diagnostics import combo_diagnostics
from optuna_tuning import maybe_tune_models
from model_selector import select_and_fit_model
from evaluation import fold_metrics_to_df
from acquisition import compute_acquisition_scores
from batch_selector import bucket_counts, select_batch

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def add_interaction_terms(df, interaction_terms):
    """Add interaction term columns to DataFrame."""
    if not interaction_terms:
        return df
    df = df.copy()
    for col1, col2, new_col in interaction_terms:
        if col1 in df.columns and col2 in df.columns:
            df[new_col] = df[col1].astype(float) * df[col2].astype(float)
    return df


def reinforce_combo_sampling(
    selected: pd.DataFrame,
    scored_pool: pd.DataFrame,
    valid_combos: list,
    misclassified_combos: list = None,
    lacking_combo_count: int = 3,
    misclass_combo_count: int = 3,
    max_total: int = 15,
    score_col: str = "acq_boundary",
) -> pd.DataFrame:
    """
    Add reinforcement samples for lacking or misclassified combos.
    
    Args:
        selected: Current batch selection
        scored_pool: Full scored candidate pool
        valid_combos: List of valid combo IDs
        misclassified_combos: List of combo IDs with misclassification (from holdout)
        lacking_combo_count: Samples to add per lacking combo (0 samples in batch)
        misclass_combo_count: Samples to add per misclassified combo
        max_total: Maximum total reinforcement samples
        score_col: Score column to rank candidates
    
    Returns:
        Combined DataFrame with original + reinforcement samples
    """
    if misclassified_combos is None:
        misclassified_combos = []
    
    # Identify lacking combos (0 samples in current batch)
    current_combos = set(selected["discrete_combo_id"].unique())
    all_combos = set(c["combo_id"] for c in valid_combos)
    lacking_combos = list(all_combos - current_combos)
    
    print(f"[INFO] Combo reinforcement: lacking={len(lacking_combos)}, misclassified={len(misclassified_combos)}")
    
    # Build exclusion set (already selected samples)
    selected_keys = set(zip(
        selected["A_Cell_D"].round(6), 
        selected["C_Barrier_Thx"].round(6), 
        selected["discrete_combo_id"]
    ))
    
    def is_already_selected(row):
        key = (round(row["A_Cell_D"], 6), round(row["C_Barrier_Thx"], 6), row["discrete_combo_id"])
        return key in selected_keys
    
    available_pool = scored_pool[~scored_pool.apply(is_already_selected, axis=1)].copy()
    
    additions = []
    total_added = 0
    
    # 1. Add samples for lacking combos
    for combo_id in lacking_combos:
        if total_added >= max_total:
            break
        combo_pool = available_pool[available_pool["discrete_combo_id"] == combo_id]
        n_add = min(lacking_combo_count, max_total - total_added, len(combo_pool))
        if n_add > 0:
            top_n = combo_pool.nlargest(n_add, score_col).copy()
            top_n["selected_bucket"] = "combo_reinforce_lacking"
            additions.append(top_n)
            total_added += n_add
            # Update available pool
            available_pool = available_pool[~available_pool.index.isin(top_n.index)]
    
    # 2. Add samples for misclassified combos
    for combo_id in misclassified_combos:
        if total_added >= max_total:
            break
        # Misclassified combos may already have some samples, add more
        combo_pool = available_pool[available_pool["discrete_combo_id"] == combo_id]
        n_add = min(misclass_combo_count, max_total - total_added, len(combo_pool))
        if n_add > 0:
            top_n = combo_pool.nlargest(n_add, score_col).copy()
            top_n["selected_bucket"] = "combo_reinforce_misclass"
            additions.append(top_n)
            total_added += n_add
            available_pool = available_pool[~available_pool.index.isin(top_n.index)]
    
    if additions:
        reinforce_df = pd.concat(additions, ignore_index=True)
        print(f"[INFO] Added {len(reinforce_df)} reinforcement samples:")
        print(f"  - Lacking combos: {reinforce_df['selected_bucket'].eq('combo_reinforce_lacking').sum()}")
        print(f"  - Misclass combos: {reinforce_df['selected_bucket'].eq('combo_reinforce_misclass').sum()}")
        print(f"  - By combo: {reinforce_df['discrete_combo_id'].value_counts().to_dict()}")
        
        # Combine and re-rank
        combined = pd.concat([selected, reinforce_df], ignore_index=True)
        combined["sampling_rank"] = range(1, len(combined) + 1)
        return combined
    else:
        print("[INFO] No reinforcement samples needed.")
        return selected


def create_try_dir(output_dir):
    pattern = re.compile(r"^Try_(\d+)$")
    existing = []
    for p in Path(output_dir).iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                existing.append(int(m.group(1)))
    next_n = 1 if not existing else max(existing) + 1
    try_dir = Path(output_dir) / f"Try_{next_n}"
    try_dir.mkdir(parents=True, exist_ok=False)
    return try_dir


def normalized_range_stats(df, cfg):
    vals = []
    for col in cfg.CONTINUOUS_COLS:
        lo, hi = cfg.CONTINUOUS_BOUNDS[col]
        denom = float(hi) - float(lo)
        if denom <= 0:
            vals.append(0.0)
            continue
        rng = float(df[col].max()) - float(df[col].min())
        vals.append(max(0.0, min(1.0, rng / denom)))
    return {
        "min_norm_range": float(min(vals)) if vals else 0.0,
        "mean_norm_range": float(np.mean(vals)) if vals else 0.0,
    }


def make_effective_boundary_weights(cfg, model_kind, labeled_df):
    use_adaptive = bool(getattr(cfg, "ENABLE_ADAPTIVE_BOUNDARY_HYBRID", False))
    base = dict(cfg.BOUNDARY_WEIGHTS_MLP if str(model_kind).lower() == "mlp" else cfg.BOUNDARY_WEIGHTS_GP)
    if not use_adaptive:
        return base

    max_unc = float(getattr(cfg, "ADAPTIVE_BOUNDARY_CLF_UNC_MAX", 0.10))
    min_unc = float(getattr(cfg, "ADAPTIVE_BOUNDARY_CLF_UNC_MIN", 0.00))
    full_min = float(
        getattr(
            cfg,
            "ADAPTIVE_BOUNDARY_COVERAGE_FULL_MIN_NORM_RANGE",
            getattr(cfg, "ADAPTIVE_BOUNDARY_COVERAGE_GUARD_MIN_NORM_RANGE", 0.90),
        )
    )
    full_mean = float(
        getattr(
            cfg,
            "ADAPTIVE_BOUNDARY_COVERAGE_FULL_MEAN_NORM_RANGE",
            getattr(cfg, "ADAPTIVE_BOUNDARY_COVERAGE_GUARD_MEAN_NORM_RANGE", 0.93),
        )
    )
    unc_span = max(0.0, max_unc - min_unc)

    cov = normalized_range_stats(labeled_df, cfg)
    min_readiness = 1.0 if full_min <= 0 else cov["min_norm_range"] / full_min
    mean_readiness = 1.0 if full_mean <= 0 else cov["mean_norm_range"] / full_mean
    readiness = max(0.0, min(1.0, min(min_readiness, mean_readiness)))
    target_unc = min_unc + unc_span * readiness

    keys_non_unc = ["boundary", "local_sparsity", "combo_priority"]
    base_non_unc_sum = sum(float(base.get(k, 0.0)) for k in keys_non_unc)
    remain = max(0.0, 1.0 - target_unc)
    if base_non_unc_sum <= 1e-12:
        eff = {
            "boundary": remain,
            "clf_uncertainty": target_unc,
            "local_sparsity": 0.0,
            "combo_priority": 0.0,
        }
    else:
        eff = {k: remain * float(base.get(k, 0.0)) / base_non_unc_sum for k in keys_non_unc}
        eff["clf_uncertainty"] = target_unc

    logging.info(
        "Adaptive boundary weights | model=%s readiness=%.3f cov_min=%.3f cov_mean=%.3f -> %s",
        model_kind,
        readiness,
        cov["min_norm_range"],
        cov["mean_norm_range"],
        {k: round(v, 4) for k, v in eff.items()},
    )
    return eff


def build_scoring_config(cfg, model_kind, effective_boundary_weights):
    d = {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
    out = SimpleNamespace(**d)
    out.BOUNDARY_WEIGHTS_GP = dict(cfg.BOUNDARY_WEIGHTS_GP)
    out.BOUNDARY_WEIGHTS_MLP = dict(cfg.BOUNDARY_WEIGHTS_MLP)
    if str(model_kind).lower() == "mlp":
        out.BOUNDARY_WEIGHTS_MLP.update(effective_boundary_weights)
    else:
        out.BOUNDARY_WEIGHTS_GP.update(effective_boundary_weights)
    return out


def save_selected_overlay_plot(base_df, selected_df, x_col, y_col, output_png, title=None):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.scatter(base_df[x_col], base_df[y_col], s=24, c="#B0B7C3", alpha=0.55, label=f"Existing (n={len(base_df)})")

    # Color selected points by bucket to make exploration/exploitation mix explicit.
    present_buckets = list(pd.unique(selected_df["selected_bucket"].dropna()))
    preferred = [b for b in config.BUCKET_RATIO.keys() if b in present_buckets]
    extras = [b for b in present_buckets if b not in preferred]
    bucket_order = preferred + extras
    cmap = plt.get_cmap("tab10")
    for i, bucket in enumerate(bucket_order):
        part = selected_df[selected_df["selected_bucket"] == bucket]
        ax.scatter(
            part[x_col],
            part[y_col],
            s=80,
            color=cmap(i % getattr(cmap, "N", 10)),
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            label=f"{bucket} (n={len(part)})",
        )

    for _, row in selected_df.iterrows():
        ax.text(row[x_col], row[y_col], str(int(row["sampling_rank"])), fontsize=7, color="#1f2937", alpha=0.8)
    ax.set_title(title or f"Selected Points on {x_col} vs {y_col}", fontsize=15, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def save_all_continuous_pair_plots(base_df, selected_df, continuous_cols, output_dir):
    written = []
    for x_col, y_col in itertools.combinations(continuous_cols, 2):
        if x_col == "A_Cell_D" and y_col == "C_Barrier_Thx":
            out_name = "sampling_cellD_vs_barrierThx.png"
            title = "Selected Points on Cell_D vs Barrier_Thx"
        else:
            x_slug = x_col.lower().replace("_", "")
            y_slug = y_col.lower().replace("_", "")
            out_name = f"sampling_{x_slug}_vs_{y_slug}.png"
            title = None
        out_path = Path(output_dir) / out_name
        save_selected_overlay_plot(base_df, selected_df, x_col, y_col, out_path, title=title)
        written.append(out_path)
    return written


def save_continuous_pairs_dashboard(base_df, selected_df, continuous_cols, output_png):
    pairs = list(itertools.combinations(continuous_cols, 2))
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), dpi=170)
    axes = axes.flatten()

    present_buckets = list(pd.unique(selected_df["selected_bucket"].dropna()))
    preferred = [b for b in config.BUCKET_RATIO.keys() if b in present_buckets]
    extras = [b for b in present_buckets if b not in preferred]
    bucket_order = preferred + extras
    cmap = plt.get_cmap("tab10")

    for ax, (x_col, y_col) in zip(axes, pairs):
        ax.scatter(base_df[x_col], base_df[y_col], s=14, c="#B0B7C3", alpha=0.45, label=f"Existing (n={len(base_df)})")

        for i, bucket in enumerate(bucket_order):
            part = selected_df[selected_df["selected_bucket"] == bucket]
            ax.scatter(
                part[x_col],
                part[y_col],
                s=55,
                color=cmap(i % getattr(cmap, "N", 10)),
                edgecolors="black",
                linewidths=0.4,
                alpha=0.92,
                label=f"{bucket} (n={len(part)})",
            )

        for _, row in selected_df.iterrows():
            ax.text(row[x_col], row[y_col], str(int(row["sampling_rank"])), fontsize=6, color="#1f2937", alpha=0.75)

        ax.set_title(f"{x_col} vs {y_col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=15)
        ax.set_ylabel(y_col, fontsize=15)
        ax.grid(True, alpha=0.22)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

    handles, labels = axes[0].get_legend_handles_labels()
    # Reserve enough room so bottom-row x-axis labels do not overlap the legend.
    fig.subplots_adjust(bottom=0.10, top=0.93)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        bbox_transform=fig.transFigure,
        frameon=True,
        title="Bucket",
        fontsize=15,
        ncol=min(5, len(labels)),
    )
    fig.suptitle("Selected Samples Across All Continuous-Pair Projections", fontsize=18, fontweight="bold", y=0.995)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_selection_dashboard(base_df, selected_df, diag_df, cfg, output_png):
    batch_size = int(cfg.BATCH_SIZE)
    ratios = dict(cfg.BUCKET_RATIO)
    ratio_total = sum(ratios.values())
    raw = {k: batch_size * v / ratio_total for k, v in ratios.items()}
    target_floor = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = batch_size - sum(target_floor.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - target_floor[k], reverse=True)
    for k in order[:rem]:
        target_floor[k] += 1
    actual_counts = selected_df["selected_bucket"].value_counts().to_dict()
    buckets = list(ratios.keys())
    actual = [actual_counts.get(b, 0) for b in buckets]
    target = [target_floor[b] for b in buckets]

    feat_cols = cfg.CONTINUOUS_COLS + cfg.DISCRETE_COLS
    x_all = pd.concat([
        base_df[feat_cols].assign(_group="existing"),
        selected_df[feat_cols].assign(_group="selected"),
    ], ignore_index=True)

    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer([
        ("cont", StandardScaler(), cfg.CONTINUOUS_COLS),
        ("disc", enc, cfg.DISCRETE_COLS),
    ])
    pcs = PCA(n_components=2, random_state=cfg.RANDOM_SEED).fit_transform(pre.fit_transform(x_all[feat_cols]))
    x_all["PC1"] = pcs[:, 0]
    x_all["PC2"] = pcs[:, 1]

    sel_combo = selected_df["discrete_combo_id"].value_counts()
    diag_plot = diag_df[["discrete_combo_id", "n_total"]].copy()
    diag_plot["selected_new"] = diag_plot["discrete_combo_id"].map(sel_combo).fillna(0).astype(int)
    diag_plot = diag_plot.sort_values("n_total")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 10), dpi=150)
    n_selected = int(len(selected_df))
    fig.suptitle(f"Active Sampling {n_selected}-Point Selection Dashboard", fontsize=22, fontweight="bold", y=0.98)

    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(buckets))
    w = 0.36
    ax1.bar(x - w / 2, target, width=w, label="Target", color="#9EC5FE")
    ax1.bar(x + w / 2, actual, width=w, label="Actual", color="#2B6CB0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, rotation=15)
    ax1.set_ylabel("Count")
    ax1.set_title("1) Bucket Allocation: Target vs Actual")
    for i, v in enumerate(actual):
        ax1.text(i + w / 2, v + 0.15, str(v), ha="center", fontsize=10)
    ax1.legend(frameon=True)

    ax2 = plt.subplot(2, 2, 2)
    ex = x_all[x_all["_group"] == "existing"]
    se = x_all[x_all["_group"] == "selected"].copy()
    se["selected_bucket"] = selected_df["selected_bucket"].to_numpy()
    se["sampling_rank"] = selected_df["sampling_rank"].to_numpy()
    ax2.scatter(ex["PC1"], ex["PC2"], s=28, c="#B0B7C3", alpha=0.55, label=f"Existing (n={len(ex)})")

    present_buckets = list(pd.unique(se["selected_bucket"].dropna()))
    preferred = [b for b in config.BUCKET_RATIO.keys() if b in present_buckets]
    extras = [b for b in present_buckets if b not in preferred]
    bucket_order = preferred + extras
    cmap = plt.get_cmap("tab10")
    bucket_color_map = {b: cmap(i % getattr(cmap, "N", 10)) for i, b in enumerate(bucket_order)}
    for i, bucket in enumerate(bucket_order):
        part = se[se["selected_bucket"] == bucket]
        color = bucket_color_map[bucket]
        ax2.scatter(
            part["PC1"],
            part["PC2"],
            s=90,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.95,
            label=f"{bucket} (n={len(part)})",
        )
        short = bucket.replace("_", "")[:5]
        for _, row in part.iterrows():
            ax2.text(row["PC1"], row["PC2"], f"{int(row['sampling_rank'])}:{short}", fontsize=6.5, color="#1f2937", alpha=0.9)

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("2) Distribution Coverage (PCA View)")
    ax2.legend(frameon=True, loc="best", fontsize=8, title="Bucket")

    ax3 = plt.subplot(2, 2, 3)
    idx = np.arange(len(diag_plot))
    ax3.bar(idx, diag_plot["n_total"], label="Before", color="#CFE8CF")
    ax3.bar(idx, diag_plot["selected_new"], bottom=diag_plot["n_total"], label=f"Selected +{n_selected}", color="#2A9D8F")
    ax3.set_title("3) Discrete Combo Coverage (Before vs After)")
    ax3.set_ylabel("Samples per combo")
    step = max(1, len(diag_plot) // min(20, len(diag_plot)))
    xt = idx[::step]
    ax3.set_xticks(xt)
    ax3.set_xticklabels(diag_plot["discrete_combo_id"].iloc[::step], rotation=70, fontsize=8)
    ax3.legend(frameon=True)

    ax4 = plt.subplot(2, 2, 4)
    plot_df = selected_df.copy()
    plot_df["_size"] = 120 + 280 * (plot_df["local_sparsity"].fillna(0).clip(0, 1).to_numpy())
    for bucket in bucket_order:
        part = plot_df[plot_df["selected_bucket"] == bucket]
        if len(part) == 0:
            continue
        ax4.scatter(
            part["p_tp"],
            part["tmax_pred_given_notp"],
            s=part["_size"],
            color=bucket_color_map[bucket],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label=f"{bucket} (n={len(part)})",
        )
    ax4.set_title("4) Selected Points: Risk-Reward Map")
    ax4.set_xlabel("Predicted TP probability (p_tp)")
    ax4.set_ylabel("Predicted Tmax given NoTP")
    ax4.legend(title="Bucket", loc="best", frameon=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def save_model_compare_cv_barplot(selection_report, output_png):
    gp = selection_report.get("gp_cv_result") or {}
    mlp = selection_report.get("mlp_cv_result") or {}
    if not gp or not mlp:
        return False

    gp_holdout = selection_report.get("gp_holdout_result") or {}
    mlp_holdout = selection_report.get("mlp_holdout_result") or {}
    has_holdout = ("error" not in gp_holdout) and ("error" not in mlp_holdout) and gp_holdout and mlp_holdout

    metrics = [
        ("tp_recall", "TP recall", True),
        ("tp_f1", "TP F1", True),
        ("stable_score", "Stable score", True),
        ("tmax_rmse", "Tmax RMSE", False),
        ("tmax_r2", "Tmax R2", True),
    ]

    labels = []
    gp_vals = []
    gp_err = []
    mlp_vals = []
    mlp_err = []
    keep_higher = []

    for key, label, higher_is_better in metrics:
        g = gp.get(f"{key}_mean", gp.get(key, np.nan))
        m = mlp.get(f"{key}_mean", mlp.get(key, np.nan))
        if not np.isfinite(g) and not np.isfinite(m):
            continue
        labels.append(label)
        gp_vals.append(float(g) if np.isfinite(g) else np.nan)
        mlp_vals.append(float(m) if np.isfinite(m) else np.nan)
        gstd = gp.get(f"{key}_std", np.nan)
        mstd = mlp.get(f"{key}_std", np.nan)
        gp_err.append(float(gstd) if np.isfinite(gstd) else 0.0)
        mlp_err.append(float(mstd) if np.isfinite(mstd) else 0.0)
        keep_higher.append(bool(higher_is_better))

    if not labels:
        return False

    x = np.arange(len(labels))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=170)
    ax = axes[0]
    bars_gp = ax.bar(x - w / 2, gp_vals, width=w, yerr=gp_err, capsize=3, label="GP", color="#4C78A8")
    bars_mlp = ax.bar(x + w / 2, mlp_vals, width=w, yerr=mlp_err, capsize=3, label="MLP", color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("CV score")
    ax.set_title("CV Metrics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)

    for i, (g, m, higher_is_better) in enumerate(zip(gp_vals, mlp_vals, keep_higher)):
        if not (np.isfinite(g) and np.isfinite(m)):
            continue
        if higher_is_better:
            winner = "GP" if g >= m else "MLP"
            y = max(g, m)
        else:
            winner = "GP" if g <= m else "MLP"
            y = max(g, m)
        ax.text(i, y, winner, ha="center", va="bottom", fontsize=8, color="#374151")

    for bars in [bars_gp, bars_mlp]:
        for b in bars:
            v = b.get_height()
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax2 = axes[1]
    if has_holdout:
        summary_labels = ["CV stable", "Holdout weighted", "Composite"]
        cv_gp = float(gp.get("stable_score", np.nan))
        cv_mlp = float(mlp.get("stable_score", np.nan))
        ho_gp = float(gp_holdout.get("weighted_score", np.nan))
        ho_mlp = float(mlp_holdout.get("weighted_score", np.nan))
        comp_gp = float(selection_report.get("gp_composite_score", np.nan))
        comp_mlp = float(selection_report.get("mlp_composite_score", np.nan))
        xx = np.arange(len(summary_labels))
        bars2_gp = ax2.bar(xx - w / 2, [cv_gp, ho_gp, comp_gp], width=w, label="GP", color="#4C78A8")
        bars2_mlp = ax2.bar(xx + w / 2, [cv_mlp, ho_mlp, comp_mlp], width=w, label="MLP", color="#F58518")
        ax2.set_xticks(xx)
        ax2.set_xticklabels(summary_labels)
        cw = selection_report.get("composite_weights", {})
        ax2.set_title(f"Summary Scores (composite: CV {cw.get('cv', 0):.2f}, Holdout {cw.get('holdout', 0):.2f})")
        ax2.set_ylabel("Score")
        ax2.grid(True, axis="y", alpha=0.25)
        ax2.legend(frameon=True)
        for bars in [bars2_gp, bars2_mlp]:
            for b in bars:
                v = b.get_height()
                if np.isfinite(v):
                    ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Holdout comparison unavailable", ha="center", va="center", fontsize=12, color="#6b7280")

    fig.suptitle("GP vs MLP Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return True

def save_holdout_confusion_matrix(y_true, y_pred, output_png, cfg):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    cm = confusion_matrix(y_true, y_pred, labels=[cfg.NOTP_LABEL, cfg.TP_LABEL])
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=cfg.TP_LABEL, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=cfg.TP_LABEL, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=cfg.TP_LABEL, zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NoTP (pred)", "TP (pred)"], fontsize=12)
    ax.set_yticklabels(["NoTP (actual)", "TP (actual)"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, fontweight="bold", color=color)

    ax.set_title("TP/NoTP Confusion Matrix (Holdout)", fontsize=15, fontweight="bold", pad=12)
    metrics_text = f"Acc: {acc:.3f}   Precision: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}"
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=11, color="#374151")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_holdout_tmax_actual_vs_pred(y_true, y_pred, output_png):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return None

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp) if len(yt) > 1 else np.nan

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(yt, yp, s=50, alpha=0.7, edgecolors="black", linewidths=0.4, c="#4C78A8")
    lims = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
    ax.plot(lims, lims, "--", color="#E45756", linewidth=1.5, label="Ideal (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Tmax", fontsize=13)
    ax.set_ylabel("Predicted Tmax", fontsize=13)
    ax.set_title("Tmax Actual vs Predicted (Holdout, NoTP only)", fontsize=14, fontweight="bold")
    metrics_text = f"MAE: {mae:.2f}   RMSE: {rmse:.2f}   R²: {r2:.3f}   N: {len(yt)}"
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=11, color="#374151")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(yt)}


def collect_iteration_history(output_dir):
    """Scan Itr_n folders and collect iteration_summary.json from each.
    Falls back to model_selection_report.json for older iterations without summary.
    """
    itr_pattern = re.compile(r"^Itr_(\d+)$")
    rows = []
    for p in Path(output_dir).iterdir():
        if not p.is_dir():
            continue
        m = itr_pattern.match(p.name)
        if not m:
            continue
        itr_num = int(m.group(1))

        # Try iteration_summary.json first (new format)
        summary_path = p / "iteration_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["itr_num"] = itr_num
                data["itr_folder"] = p.name
                rows.append(data)
                continue
            except Exception:
                pass

        # Fallback: try model_selection_report.json for older iterations
        report_path = p / "model_selection_report.json"
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                data = {
                    "itr_num": itr_num,
                    "itr_folder": p.name,
                    "selected_model": report.get("selected_model", "gp"),
                    "gp_score": report.get("gp_score"),
                    "mlp_score": report.get("mlp_score"),
                    "gp_composite_score": report.get("gp_composite_score"),
                    "mlp_composite_score": report.get("mlp_composite_score"),
                }
                rows.append(data)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("itr_num").reset_index(drop=True)
    return df


def save_iteration_performance_trend_plot(history_df, output_png):
    """Plot performance metrics trend across iterations."""
    if len(history_df) < 1:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    x = history_df["itr_num"].to_numpy()

    def _has_valid_data(df, col):
        return col in df.columns and df[col].notna().any()

    # Panel 1: TP Recall & F1
    ax1 = axes[0, 0]
    has_p1_data = False
    if _has_valid_data(history_df, "holdout_recall"):
        ax1.plot(x, history_df["holdout_recall"], marker="o", linewidth=2, label="TP Recall", color="#4C78A8")
        has_p1_data = True
    if _has_valid_data(history_df, "holdout_f1"):
        ax1.plot(x, history_df["holdout_f1"], marker="s", linewidth=2, label="TP F1", color="#F58518")
        has_p1_data = True
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Score")
    ax1.set_title("Holdout Classification: TP Recall & F1")
    if has_p1_data:
        ax1.legend(frameon=True)
    else:
        ax1.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax1.transAxes, fontsize=12, color="gray")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)

    # Panel 2: Accuracy & Precision
    ax2 = axes[0, 1]
    has_p2_data = False
    if _has_valid_data(history_df, "holdout_accuracy"):
        ax2.plot(x, history_df["holdout_accuracy"], marker="o", linewidth=2, label="Accuracy", color="#54A24B")
        has_p2_data = True
    if _has_valid_data(history_df, "holdout_precision"):
        ax2.plot(x, history_df["holdout_precision"], marker="s", linewidth=2, label="Precision", color="#E45756")
        has_p2_data = True
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Score")
    ax2.set_title("Holdout Classification: Accuracy & Precision")
    if has_p2_data:
        ax2.legend(frameon=True)
    else:
        ax2.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax2.transAxes, fontsize=12, color="gray")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)

    # Panel 3: Tmax RMSE & R2
    ax3 = axes[1, 0]
    has_p3_data = False
    if _has_valid_data(history_df, "holdout_tmax_rmse"):
        vals = history_df["holdout_tmax_rmse"].to_numpy()
        if np.isfinite(vals).any():
            ax3.plot(x, vals, marker="o", linewidth=2, label="Tmax RMSE", color="#B279A2")
            has_p3_data = True
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("RMSE")
    ax3.set_title("Holdout Regression: Tmax RMSE")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x)

    ax3b = ax3.twinx()
    has_p3b_data = False
    if _has_valid_data(history_df, "holdout_tmax_r2"):
        vals = history_df["holdout_tmax_r2"].to_numpy()
        if np.isfinite(vals).any():
            ax3b.plot(x, vals, marker="s", linewidth=2, linestyle="--", label="Tmax R²", color="#72B7B2")
            has_p3b_data = True
    ax3b.set_ylabel("R²")
    if has_p3_data or has_p3b_data:
        handles1, labels1 = ax3.get_legend_handles_labels()
        handles2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(handles1 + handles2, labels1 + labels2, frameon=True, loc="best")
    if not has_p3_data and not has_p3b_data:
        ax3.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax3.transAxes, fontsize=12, color="gray")

    # Panel 4: Sampling uncertainty trend (Option B)
    ax4 = axes[1, 1]
    has_p4_left_data = False
    if _has_valid_data(history_df, "uncertainty_mean"):
        ax4.plot(x, history_df["uncertainty_mean"], marker="o", linewidth=2, label="Uncertainty Mean", color="#4C78A8")
        has_p4_left_data = True
    if _has_valid_data(history_df, "uncertainty_p90"):
        ax4.plot(x, history_df["uncertainty_p90"], marker="s", linewidth=2, linestyle="--", label="Uncertainty P90", color="#F58518")
        has_p4_left_data = True

    ax4b = ax4.twinx()
    has_p4_right_data = False
    if _has_valid_data(history_df, "uncertainty_high_ratio"):
        ax4b.plot(x, history_df["uncertainty_high_ratio"], marker="^", linewidth=2, linestyle=":", label="High-Uncertainty Ratio", color="#E45756")
        has_p4_right_data = True

    if has_p4_left_data or has_p4_right_data:
        handles1, labels1 = ax4.get_legend_handles_labels()
        handles2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(handles1 + handles2, labels1 + labels2, frameon=True, loc="best")
    else:
        ax4.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax4.transAxes, fontsize=12, color="gray")

    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Uncertainty")
    ax4b.set_ylabel("High Ratio")
    ax4.set_title("Sampling Uncertainty: Mean, P90, High Ratio")
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(x)

    fig.suptitle("Iteration Performance Trend (Holdout)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return True


def recommend_next_batch_size(history_df, cfg):
    """Recommend next batch size based on recent iteration performance."""
    window = int(getattr(cfg, "BATCH_SIZE_DECISION_WINDOW", 3))
    step_up = int(getattr(cfg, "BATCH_SIZE_STEP_UP", 5))
    step_down = int(getattr(cfg, "BATCH_SIZE_STEP_DOWN", 5))
    min_batch = int(getattr(cfg, "BATCH_SIZE_MIN", 16))
    max_batch = int(getattr(cfg, "BATCH_SIZE_MAX", 40))
    current = int(cfg.BATCH_SIZE)

    if len(history_df) < window:
        return {
            "current_batch_size": current,
            "recommended_batch_size": current,
            "decision": "hold",
            "reason": f"Not enough iterations ({len(history_df)} < {window}) for decision.",
        }

    recent = history_df.tail(window)

    # Check recall trend
    recall_vals = recent["holdout_recall"].dropna().to_numpy() if "holdout_recall" in recent.columns else np.array([])
    f1_vals = recent["holdout_f1"].dropna().to_numpy() if "holdout_f1" in recent.columns else np.array([])
    rmse_vals = recent["holdout_tmax_rmse"].dropna().to_numpy() if "holdout_tmax_rmse" in recent.columns else np.array([])

    recall_stable = len(recall_vals) >= 2 and (recall_vals[-1] >= recall_vals[0] - 0.05)
    f1_stable = len(f1_vals) >= 2 and (f1_vals[-1] >= f1_vals[0] - 0.05)
    rmse_stable = len(rmse_vals) < 2 or (rmse_vals[-1] <= rmse_vals[0] * 1.1)

    recall_improving = len(recall_vals) >= 2 and (recall_vals[-1] > recall_vals[0] + 0.02)
    f1_improving = len(f1_vals) >= 2 and (f1_vals[-1] > f1_vals[0] + 0.02)

    recall_degrading = len(recall_vals) >= 2 and (recall_vals[-1] < recall_vals[0] - 0.05)
    f1_degrading = len(f1_vals) >= 2 and (f1_vals[-1] < f1_vals[0] - 0.05)
    rmse_degrading = len(rmse_vals) >= 2 and (rmse_vals[-1] > rmse_vals[0] * 1.15)

    if recall_degrading or f1_degrading or rmse_degrading:
        new_batch = max(min_batch, current - step_down)
        return {
            "current_batch_size": current,
            "recommended_batch_size": new_batch,
            "decision": "decrease",
            "reason": f"Performance degrading (recall_deg={recall_degrading}, f1_deg={f1_degrading}, rmse_deg={rmse_degrading}).",
        }

    if recall_stable and f1_stable and rmse_stable:
        if recall_improving or f1_improving:
            new_batch = min(max_batch, current + step_up)
            return {
                "current_batch_size": current,
                "recommended_batch_size": new_batch,
                "decision": "increase",
                "reason": f"Performance stable and improving (recall_impr={recall_improving}, f1_impr={f1_improving}).",
            }
        else:
            return {
                "current_batch_size": current,
                "recommended_batch_size": current,
                "decision": "hold",
                "reason": "Performance stable but not significantly improving.",
            }

    return {
        "current_batch_size": current,
        "recommended_batch_size": current,
        "decision": "hold",
        "reason": "Mixed signals; holding current batch size.",
    }


def normalize_ratio_dict(ratio_dict):
    """Normalize ratio dict so values sum to 1.0; fallback to uniform."""
    keys = list(ratio_dict.keys())
    if not keys:
        return {}
    total = sum(float(v) for v in ratio_dict.values())
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: float(ratio_dict.get(k, 0.0)) / total for k in keys}


def clamp_ratio_dict(ratio_dict, min_dict, max_dict):
    """Clamp each ratio within bounds and normalize."""
    out = {}
    for k, v in ratio_dict.items():
        lo = float(min_dict.get(k, 0.0))
        hi = float(max_dict.get(k, 1.0))
        out[k] = min(max(float(v), lo), hi)
    return normalize_ratio_dict(out)


def recommend_bucket_ratio(history_df, cfg, current_ratio):
    """Recommend bucket ratio from recent holdout/uncertainty trends."""
    window = int(getattr(cfg, "BUCKET_RATIO_DECISION_WINDOW", 3))
    step_base = float(getattr(cfg, "BUCKET_RATIO_STEP_BASE", 0.08))
    step_max = float(getattr(cfg, "BUCKET_RATIO_STEP_MAX", 0.12))
    step_stable = float(getattr(cfg, "BUCKET_RATIO_STEP_STABLE", 0.04))
    unc_target_low = float(getattr(cfg, "UNCERTAINTY_TARGET_LOW", 0.20))
    unc_target_high = float(getattr(cfg, "UNCERTAINTY_TARGET_HIGH", 0.35))
    min_dict = dict(getattr(cfg, "BUCKET_RATIO_MIN", {}))
    max_dict = dict(getattr(cfg, "BUCKET_RATIO_MAX", {}))

    current = normalize_ratio_dict(dict(current_ratio))
    if len(history_df) < window:
        return {
            "current_bucket_ratio": {k: float(v) for k, v in current.items()},
            "recommended_bucket_ratio": {k: float(v) for k, v in current.items()},
            "decision": "hold",
            "reason": f"Not enough iterations ({len(history_df)} < {window}) for decision.",
        }

    recent = history_df.tail(window)
    f1_vals = recent["holdout_f1"].dropna().to_numpy() if "holdout_f1" in recent.columns else np.array([])
    recall_vals = recent["holdout_recall"].dropna().to_numpy() if "holdout_recall" in recent.columns else np.array([])
    rmse_vals = recent["holdout_tmax_rmse"].dropna().to_numpy() if "holdout_tmax_rmse" in recent.columns else np.array([])
    unc_vals = recent["uncertainty_high_ratio"].dropna().to_numpy() if "uncertainty_high_ratio" in recent.columns else np.array([])

    f1_delta = float(f1_vals[-1] - f1_vals[0]) if len(f1_vals) >= 2 else 0.0
    recall_delta = float(recall_vals[-1] - recall_vals[0]) if len(recall_vals) >= 2 else 0.0
    rmse_delta_pct = float((rmse_vals[-1] - rmse_vals[0]) / (rmse_vals[0] + 1e-12) * 100.0) if len(rmse_vals) >= 2 else 0.0
    unc_high_now = float(unc_vals[-1]) if len(unc_vals) >= 1 else None

    severe = (f1_delta < -0.04) or (recall_delta < -0.04) or (rmse_delta_pct > 10.0)
    improving = ((f1_delta > 0.01) or (recall_delta > 0.01)) and (rmse_delta_pct <= 5.0)
    not_degrading = (f1_delta > -0.04) and (recall_delta > -0.04) and (rmse_delta_pct <= 5.0)

    if severe:
        step = step_max
        decision = "severe_degradation"
    elif improving and not_degrading:
        step = step_stable
        decision = "stable_improving"
    else:
        step = step_base
        decision = "mixed"

    rec = dict(current)
    if severe:
        rec["boundary"] = rec.get("boundary", 0.0) + step
        rec["uncertainty_sparse"] = rec.get("uncertainty_sparse", 0.0) - step * 0.6
        rec["random_check"] = rec.get("random_check", 0.0) - step * 0.4

    if unc_high_now is not None and unc_high_now > unc_target_high:
        rec["uncertainty_sparse"] = rec.get("uncertainty_sparse", 0.0) - step * 0.5
        rec["boundary"] = rec.get("boundary", 0.0) + step * 0.3
    elif unc_high_now is not None and unc_high_now < unc_target_low and not severe:
        rec["uncertainty_sparse"] = rec.get("uncertainty_sparse", 0.0) + step * 0.4
        rec["boundary"] = rec.get("boundary", 0.0) - step * 0.2

    if rmse_delta_pct > 5.0:
        rec["notp_high_tmax"] = rec.get("notp_high_tmax", 0.0) + step * 0.3
        rec["random_check"] = rec.get("random_check", 0.0) - step * 0.2

    rec = clamp_ratio_dict(rec, min_dict, max_dict)

    unc_txt = "N/A" if unc_high_now is None else f"{unc_high_now:.3f}"
    reason = (
        f"decision={decision}; f1_delta={f1_delta:.4f}; recall_delta={recall_delta:.4f}; "
        f"rmse_delta_pct={rmse_delta_pct:.2f}; unc_high={unc_txt}"
    )

    return {
        "current_bucket_ratio": {k: float(v) for k, v in current.items()},
        "recommended_bucket_ratio": {k: float(v) for k, v in rec.items()},
        "decision": decision,
        "reason": reason,
    }


def _count_from_ratio(total, labels, ratio_by_label):
    raw = {k: total * float(ratio_by_label.get(k, 0.0)) for k in labels}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = int(total) - sum(counts.values())
    order = sorted(labels, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:rem]:
        counts[k] += 1
    return counts

def build_tp1_ratio_celld_quota(df, cfg, bucket_target_n):
    if not bool(getattr(cfg, "NOTP_HIGHTMAX_USE_TP1_CELLD_RATIO", False)):
        return None
    col = getattr(cfg, "TP_RATIO_CELLD_COL", getattr(cfg, "NOTP_HIGHTMAX_CELLD_COL", "A_Cell_D"))
    bins = list(getattr(cfg, "TP_RATIO_CELLD_BINS", getattr(cfg, "NOTP_HIGHTMAX_CELLD_BINS", [])))
    labels = list(getattr(cfg, "TP_RATIO_CELLD_BIN_LABELS", getattr(cfg, "NOTP_HIGHTMAX_CELLD_BIN_LABELS", [])))
    if len(bins) < 2 or len(labels) != len(bins) - 1:
        return None

    use = df[[col, cfg.TPNoTP_COL]].copy()
    use = use.dropna(subset=[col, cfg.TPNoTP_COL])
    tp = use[use[cfg.TPNoTP_COL] == cfg.TP_LABEL]
    if len(tp) == 0:
        ratio = {k: 1.0 / len(labels) for k in labels}
    else:
        cat = pd.cut(tp[col], bins=bins, labels=labels, include_lowest=True, right=False)
        cnt = tp.groupby(cat, observed=False).size().reindex(labels, fill_value=0)
        total = int(cnt.sum())
        ratio = {k: (float(cnt.loc[k]) / total if total > 0 else 1.0 / len(labels)) for k in labels}
        if sum(ratio.values()) <= 0:
            ratio = {k: 1.0 / len(labels) for k in labels}

    quota = _count_from_ratio(int(bucket_target_n), labels, ratio)
    return {
        "col": col,
        "bins": bins,
        "labels": labels,
        "quota_by_label": quota,
        "ratio_by_label": ratio,
    }

def build_notp_high_tmax_celld_quota(df, cfg, bucket_target_n):
    # Backward-compatible wrapper used by previous ad-hoc analysis snippets.
    return build_tp1_ratio_celld_quota(df, cfg, bucket_target_n)

def build_fixed_zone_quota(cfg, target_n, ratio_dict):
    col = getattr(cfg, "HYBRID_CELLD_COL", "A_Cell_D")
    bins = list(getattr(cfg, "HYBRID_CELLD_BINS", []))
    labels = list(getattr(cfg, "HYBRID_CELLD_BIN_LABELS", []))
    if len(bins) < 2 or len(labels) != len(bins) - 1:
        return None
    ratio = {k: float(ratio_dict.get(k, 0.0)) for k in labels}
    s = sum(ratio.values())
    if s <= 0:
        ratio = {k: 1.0 / len(labels) for k in labels}
    else:
        ratio = {k: v / s for k, v in ratio.items()}
    quota = _count_from_ratio(int(target_n), labels, ratio)
    return {
        "col": col,
        "bins": bins,
        "labels": labels,
        "quota_by_label": quota,
        "ratio_by_label": ratio,
    }

def build_bucket_bin_quota_rules(df, cfg, bucket_target_counts):
    mode = str(getattr(cfg, "BUCKET_CELLD_QUOTA_MODE", "off")).strip().lower()
    rules = {}

    if mode == "tp_ratio_only":
        for bucket in ["boundary", "notp_high_tmax"]:
            q = build_tp1_ratio_celld_quota(df, cfg, int(bucket_target_counts.get(bucket, 0)))
            if q is not None:
                rules[bucket] = {
                    "col": q["col"],
                    "bins": q["bins"],
                    "labels": q["labels"],
                    "quota_by_label": q["quota_by_label"],
                }
                print(f"[INFO] {bucket} TP-ratio-by-bin:")
                print(json.dumps(q["ratio_by_label"], indent=2, ensure_ascii=False))
                print(f"[INFO] {bucket} quota-by-bin:")
                print(json.dumps(q["quota_by_label"], indent=2, ensure_ascii=False))
        return rules

    if mode == "hybrid_baseline":
        q_boundary = build_fixed_zone_quota(cfg, int(bucket_target_counts.get("boundary", 0)), getattr(cfg, "HYBRID_BOUNDARY_ZONE_RATIO", {}))
        q_notp = build_fixed_zone_quota(cfg, int(bucket_target_counts.get("notp_high_tmax", 0)), getattr(cfg, "HYBRID_NOTP_HIGHTMAX_ZONE_RATIO", {}))
        if q_boundary is not None:
            rules["boundary"] = {
                "col": q_boundary["col"],
                "bins": q_boundary["bins"],
                "labels": q_boundary["labels"],
                "quota_by_label": q_boundary["quota_by_label"],
            }
            print("[INFO] boundary hybrid ratio-by-zone:")
            print(json.dumps(q_boundary["ratio_by_label"], indent=2, ensure_ascii=False))
            print("[INFO] boundary hybrid quota-by-zone:")
            print(json.dumps(q_boundary["quota_by_label"], indent=2, ensure_ascii=False))
        if q_notp is not None:
            rules["notp_high_tmax"] = {
                "col": q_notp["col"],
                "bins": q_notp["bins"],
                "labels": q_notp["labels"],
                "quota_by_label": q_notp["quota_by_label"],
            }
            print("[INFO] notp_high_tmax hybrid ratio-by-zone:")
            print(json.dumps(q_notp["ratio_by_label"], indent=2, ensure_ascii=False))
            print("[INFO] notp_high_tmax hybrid quota-by-zone:")
            print(json.dumps(q_notp["quota_by_label"], indent=2, ensure_ascii=False))
    return rules

def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try_dir = create_try_dir(config.OUTPUT_DIR)

    output_candidates_csv = try_dir / "next_sampling_candidates.csv"
    output_scored_pool_csv = try_dir / "scored_candidate_pool_preview.csv"
    output_diagnostics_csv = try_dir / "combo_diagnostics.csv"
    output_model_selection_json = try_dir / "model_selection_report.json"
    output_optuna_report_json = try_dir / "optuna_report.json"
    output_cv_fold_metrics_csv = try_dir / "cv_fold_metrics.csv"
    output_model_compare_cv_png = try_dir / "gp_vs_mlp_cv_metrics.png"
    output_pairs_dashboard_png = try_dir / "sampling_all_continuous_pairs_6panel.png"
    output_dashboard_png = try_dir / "sampling_selection_dashboard_1page.png"

    df = load_labeled_data(config.INPUT_CSV)
    df = add_interaction_terms(df, getattr(config, "INTERACTION_TERMS", []))
    validate_required_columns(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL, config.OTHER_REGRESSION_COLS, config.TIME_FEATURE_COLS)
    validate_passfail_labels(df, config.TPNoTP_COL, config.PASS_LABEL, config.FAIL_LABEL)
    valid_combos = generate_valid_discrete_combinations(config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX)
    print(f"[INFO] Valid discrete combinations: {len(valid_combos)}")
    if len(valid_combos) != 28: print("[WARN] Valid combo count is not 28. Check DISCRETE_LEVELS and constraint logic.")
    df = attach_discrete_combo_id(df, valid_combos, config.DISCRETE_COLS)
    diag = combo_diagnostics(df, valid_combos, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL, config.PASS_LABEL, config.FAIL_LABEL)
    diag.to_csv(output_diagnostics_csv, index=False, encoding="utf-8-sig")
    x_raw, y_class, y_tmax = make_xy(df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL)
    y_extra, extra_cols = make_extra_targets(df, config.OTHER_REGRESSION_COLS, config.TIME_FEATURE_COLS)
    pre = build_preprocessor(config.CONTINUOUS_COLS, config.DISCRETE_COLS)
    x_train = pre.fit_transform(x_raw)
    tuned = maybe_tune_models(df, x_train, y_class, y_tmax, y_extra, config)
    with open(output_optuna_report_json, "w", encoding="utf-8") as f:
        json.dump(tuned["report"], f, indent=2, ensure_ascii=False)
    print("[INFO] Optuna report:"); print(json.dumps(tuned["report"], indent=2, ensure_ascii=False))
    selected_model, selection_report, fold_results = select_and_fit_model(df, x_train, y_class, y_tmax, y_extra, config, tuned)
    with open(output_model_selection_json, "w", encoding="utf-8") as f:
        json.dump(selection_report, f, indent=2, ensure_ascii=False)
    fold_df = fold_metrics_to_df(fold_results)
    if len(fold_df):
        fold_df.to_csv(output_cv_fold_metrics_csv, index=False, encoding="utf-8-sig")
    wrote_model_compare_png = save_model_compare_cv_barplot(selection_report, output_model_compare_cv_png)
    if wrote_model_compare_png:
        print(f"[INFO] Saved GP vs MLP CV comparison chart: {output_model_compare_cv_png}")
    print("[INFO] Model selection report:"); print(json.dumps(selection_report, indent=2, ensure_ascii=False))
    print(f"[INFO] Selected model kind: {getattr(selected_model, 'kind', 'gp')}")
    pool = generate_candidate_pool(valid_combos, config.BASE_CONTINUOUS_COLS, config.CONTINUOUS_BOUNDS, config.DISCRETE_COLS, config.CANDIDATES_PER_COMBO, config.EXCLUDED_REFERENCE_RANGES, config.RANDOM_SEED)
    pool = add_interaction_terms(pool, getattr(config, "INTERACTION_TERMS", []))
    print(f"[INFO] Candidate pool size after exclusion filter: {len(pool)}")
    x_candidate = pre.transform(pool[config.CONTINUOUS_COLS + config.DISCRETE_COLS].copy())
    effective_boundary_weights = make_effective_boundary_weights(config, getattr(selected_model, "kind", "gp"), df)
    scoring_cfg = build_scoring_config(config, getattr(selected_model, "kind", "gp"), effective_boundary_weights)
    scored = compute_acquisition_scores(pool, df, x_candidate, x_train, selected_model, scoring_cfg)
    scored.nlargest(2000, "acq_boundary").to_csv(output_scored_pool_csv, index=False, encoding="utf-8-sig")

    # === Dynamic batch size & bucket ratio control ===
    confirmed_history_df = collect_iteration_history(config.OUTPUT_DIR)

    batch_size_mode = str(getattr(config, "BATCH_SIZE_MODE", "manual")).lower()
    bucket_ratio_mode = str(getattr(config, "BUCKET_RATIO_MODE", "manual")).lower()

    batch_rec = recommend_next_batch_size(confirmed_history_df, config)
    bucket_ratio_rec = recommend_bucket_ratio(confirmed_history_df, config, config.BUCKET_RATIO)

    if batch_size_mode == "auto" and batch_rec.get("recommended_batch_size"):
        effective_batch_size = int(batch_rec["recommended_batch_size"])
        print(f"[INFO] BATCH_SIZE_MODE=auto -> applying recommended batch_size={effective_batch_size}")
    else:
        effective_batch_size = int(config.BATCH_SIZE)
        if batch_size_mode == "shadow":
            print(f"[INFO] BATCH_SIZE_MODE=shadow -> using config batch_size={effective_batch_size}, recommended={batch_rec.get('recommended_batch_size')}")

    if bucket_ratio_mode == "auto" and bucket_ratio_rec.get("recommended_bucket_ratio"):
        effective_bucket_ratio = bucket_ratio_rec["recommended_bucket_ratio"]
        print(f"[INFO] BUCKET_RATIO_MODE=auto -> applying recommended bucket_ratio: {effective_bucket_ratio}")
    else:
        effective_bucket_ratio = dict(config.BUCKET_RATIO)
        if bucket_ratio_mode == "shadow":
            print(f"[INFO] BUCKET_RATIO_MODE=shadow -> using config bucket_ratio, recommended: {bucket_ratio_rec.get('recommended_bucket_ratio')}")

    bucket_target_counts = bucket_counts(effective_batch_size, effective_bucket_ratio)
    bucket_bin_quota_rules = build_bucket_bin_quota_rules(df, config, bucket_target_counts)

    selected = select_batch(
        scored,
        x_candidate,
        df,
        effective_batch_size,
        effective_bucket_ratio,
        config.MAX_SAMPLES_PER_COMBO,
        config.MIN_BATCH_DISTANCE,
        config.RANDOM_SEED,
        getattr(config, "BUCKET_DISTANCE_MULTIPLIER", {}),
        getattr(config, "BUCKET_LOCAL_DISTANCE_RULES", {}),
        bucket_bin_quota_rules,
        getattr(config, "BUCKET_PTP_BOUNDS", {}),
    )

    # === Combo Reinforcement Sampling ===
    if getattr(config, "ENABLE_COMBO_REINFORCE", False):
        print("[INFO] Combo reinforcement sampling enabled.")
        
        # Find previous iteration's misclassified combos
        misclassified_combos = []
        prev_itr_dirs = sorted(
            [d for d in config.OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("Itr_")],
            key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0,
            reverse=True
        )
        if prev_itr_dirs:
            prev_misclass_csv = prev_itr_dirs[0] / "Performance" / "holdout_misclassified_samples.csv"
            if prev_misclass_csv.exists():
                try:
                    prev_misclass_df = pd.read_csv(prev_misclass_csv)
                    if "discrete_combo_id" in prev_misclass_df.columns:
                        misclassified_combos = prev_misclass_df["discrete_combo_id"].unique().tolist()
                        print(f"[INFO] Found {len(misclassified_combos)} misclassified combos from {prev_misclass_csv.parent.parent.name}")
                except Exception as e:
                    print(f"[WARN] Could not read previous misclassified samples: {e}")
        
        selected = reinforce_combo_sampling(
            selected=selected,
            scored_pool=scored,
            valid_combos=valid_combos,
            misclassified_combos=misclassified_combos,
            lacking_combo_count=getattr(config, "REINFORCE_LACKING_COMBO_COUNT", 3),
            misclass_combo_count=getattr(config, "REINFORCE_MISCLASS_COMBO_COUNT", 3),
            max_total=getattr(config, "REINFORCE_MAX_TOTAL", 15),
            score_col=getattr(config, "REINFORCE_SCORE_COL", "acq_boundary"),
        )
    
    # Column order: base continuous first, interaction terms at the very end
    base_cont = getattr(config, "BASE_CONTINUOUS_COLS", config.CONTINUOUS_COLS)
    interaction_cols = [t[2] for t in getattr(config, "INTERACTION_TERMS", [])]
    front = ["sampling_rank", "selected_bucket", "selected_model_kind"] + list(base_cont) + config.DISCRETE_COLS + ["discrete_combo_id"]
    score_cols = ["p_tp", "p_notp", "boundary_score", "clf_uncertainty_raw", "clf_uncertainty_scaled", "tmax_pred_given_notp", "tmax_std_given_notp", "notp_window_score", "local_sparsity", "combo_priority", "acq_boundary", "acq_notp_high_tmax", "acq_uncertainty_sparse"]
    # Only include interaction columns that exist in the dataframe
    interaction_cols = [c for c in interaction_cols if c in selected.columns]
    selected = selected[front + score_cols + interaction_cols]
    selected.to_csv(output_candidates_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved next sampling candidates: {output_candidates_csv}")
    print(selected[["sampling_rank", "selected_bucket", "selected_model_kind", "discrete_combo_id", "p_tp", "p_notp", "tmax_pred_given_notp"]].head(20))

    save_continuous_pairs_dashboard(df, selected, config.CONTINUOUS_COLS, output_pairs_dashboard_png)
    print(f"[INFO] Saved continuous-pairs dashboard: {output_pairs_dashboard_png}")
    save_selection_dashboard(df, selected, diag, config, output_dashboard_png)
    print(f"[INFO] Saved dashboard: {output_dashboard_png}")

    # === Holdout Performance Evaluation ===
    performance_dir = try_dir / "Performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    holdout_cm_png = performance_dir / "tp_notp_holdout_confusion_matrix.png"
    holdout_tmax_png = performance_dir / "tmax_actual_vs_pred_holdout.png"

    final_test_csv = getattr(config, "FINAL_TEST_CSV", None)
    cm_metrics = None
    tmax_metrics = None
    y_holdout_class = None
    y_holdout_pred_class = None

    if final_test_csv and Path(final_test_csv).exists():
        print(f"[INFO] Loading holdout test set: {final_test_csv}")
        holdout_df = load_labeled_data(final_test_csv)
        holdout_df = add_interaction_terms(holdout_df, getattr(config, "INTERACTION_TERMS", []))
        try:
            validate_required_columns(holdout_df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL, [], [])
            holdout_df = attach_discrete_combo_id(holdout_df, valid_combos, config.DISCRETE_COLS)
            x_holdout_raw, y_holdout_class, y_holdout_tmax = make_xy(holdout_df, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL)
            x_holdout = pre.transform(x_holdout_raw)

            from acquisition import predict_outputs
            holdout_pred = predict_outputs(selected_model, x_holdout)
            y_holdout_pred_class = (np.asarray(holdout_pred["p_tp"], dtype=float) >= 0.5).astype(int)

            cm_metrics = save_holdout_confusion_matrix(y_holdout_class, y_holdout_pred_class, holdout_cm_png, config)
            print(f"[INFO] Saved holdout confusion matrix: {holdout_cm_png}")
            print(f"[INFO] Holdout classification metrics: {cm_metrics}")

            # === Save misclassified samples detail ===
            misclassified_mask = (y_holdout_class != y_holdout_pred_class)
            if misclassified_mask.sum() > 0:
                misclassified_df = holdout_df.loc[misclassified_mask].copy()
                # Add 1-based row number from original holdout CSV (header is row 1, data starts row 2)
                misclassified_df["holdout_row_num"] = [i + 2 for i in misclassified_df.index]
                misclassified_df["actual_label"] = y_holdout_class[misclassified_mask]
                misclassified_df["predicted_label"] = y_holdout_pred_class[misclassified_mask]
                misclassified_df["p_tp"] = np.asarray(holdout_pred["p_tp"], dtype=float)[misclassified_mask]
                misclassified_df["p_notp"] = np.asarray(holdout_pred["p_notp"], dtype=float)[misclassified_mask]
                misclassified_df["error_type"] = misclassified_df.apply(
                    lambda r: "FN (missed TP)" if r["actual_label"] == config.TP_LABEL else "FP (false TP)", axis=1
                )
                # Add pattern analysis
                def analyze_pattern(row):
                    notes = []
                    p_tp_val = row["p_tp"]
                    cell_d = row.get("A_Cell_D", None)
                    combo = row.get("discrete_combo_id", "")
                    barrier_type = row.get("B_Barrier_Type", "")
                    outer_type = row.get("D_Barrier_Outer_Type", "")
                    if row["error_type"] == "FP (false TP)":
                        if p_tp_val > 0.7:
                            notes.append("high confidence FP (p_tp>{:.2f})".format(p_tp_val))
                        if cell_d is not None and cell_d < 10:
                            notes.append("low Cell_D={:.1f}".format(cell_d))
                    else:  # FN
                        if 0.4 <= p_tp_val <= 0.6:
                            notes.append("boundary region (p_tp={:.2f})".format(p_tp_val))
                        elif p_tp_val < 0.4:
                            notes.append("low p_tp={:.2f}".format(p_tp_val))
                        if cell_d is not None and cell_d > 12:
                            notes.append("high Cell_D={:.1f}".format(cell_d))
                    if barrier_type and outer_type:
                        notes.append(f"combo={barrier_type}-{outer_type}")
                    return "; ".join(notes) if notes else "-"
                misclassified_df["pattern_note"] = misclassified_df.apply(analyze_pattern, axis=1)

                misclassified_csv = performance_dir / "holdout_misclassified_samples.csv"
                cols_front = ["holdout_row_num", "error_type", "pattern_note", "actual_label", "predicted_label", "p_tp", "p_notp"] + config.CONTINUOUS_COLS + config.DISCRETE_COLS
                cols_rest = [c for c in misclassified_df.columns if c not in cols_front]
                misclassified_df = misclassified_df[cols_front + cols_rest]
                misclassified_df.to_csv(misclassified_csv, index=False, encoding="utf-8-sig")
                print(f"[INFO] Saved misclassified samples: {misclassified_csv} ({len(misclassified_df)} samples)")
                fn_count = (misclassified_df["error_type"] == "FN (missed TP)").sum()
                fp_count = (misclassified_df["error_type"] == "FP (false TP)").sum()
                print(f"[INFO] Misclassified breakdown: FN={fn_count}, FP={fp_count}")

            notp_mask = (y_holdout_class == config.PASS_LABEL)
            if int(notp_mask.sum()) > 0:
                tmax_metrics = save_holdout_tmax_actual_vs_pred(
                    y_holdout_tmax[notp_mask],
                    np.asarray(holdout_pred["tmax_pred"], dtype=float)[notp_mask],
                    holdout_tmax_png,
                )
                print(f"[INFO] Saved holdout Tmax plot: {holdout_tmax_png}")
                print(f"[INFO] Holdout Tmax metrics: {tmax_metrics}")
            else:
                print("[WARN] No NoTP samples in holdout set for Tmax evaluation.")
        except Exception as e:
            print(f"[WARN] Holdout evaluation failed: {e}")
    else:
        print(f"[WARN] FINAL_TEST_CSV not set or file not found. Skipping holdout evaluation.")

    # === Save iteration summary for cumulative tracking ===
    bucket_counts_actual = selected["selected_bucket"].value_counts().to_dict() if "selected_bucket" in selected.columns else {}
    n_unique_combos = int(selected["discrete_combo_id"].nunique()) if "discrete_combo_id" in selected.columns else 0

    uncertainty_threshold = float(getattr(config, "UNCERTAINTY_HIGH_THRESHOLD", 0.70))
    uncertainty_values = np.array([], dtype=float)
    uncertainty_source = None
    if "clf_uncertainty_scaled" in selected.columns:
        uncertainty_values = pd.to_numeric(selected["clf_uncertainty_scaled"], errors="coerce").dropna().to_numpy(dtype=float)
        uncertainty_source = "clf_uncertainty_scaled"
    elif "clf_uncertainty_raw" in selected.columns:
        uncertainty_values = pd.to_numeric(selected["clf_uncertainty_raw"], errors="coerce").dropna().to_numpy(dtype=float)
        uncertainty_source = "clf_uncertainty_raw"

    uncertainty_mean = float(np.nanmean(uncertainty_values)) if uncertainty_values.size else None
    uncertainty_p90 = float(np.nanpercentile(uncertainty_values, 90)) if uncertainty_values.size else None
    uncertainty_high_ratio = float(np.mean(uncertainty_values >= uncertainty_threshold)) if uncertainty_values.size else None

    iteration_summary = {
        "input_csv": str(config.INPUT_CSV),
        "input_n": int(len(df)),
        "batch_size": int(effective_batch_size),
        "batch_size_mode": batch_size_mode,
        "batch_size_config": int(config.BATCH_SIZE),
        "batch_size_recommended": int(batch_rec.get("recommended_batch_size", config.BATCH_SIZE)) if batch_rec.get("recommended_batch_size") else None,
        "batch_size_decision": batch_rec.get("decision"),
        "bucket_ratio_mode": bucket_ratio_mode,
        "bucket_ratio_applied": effective_bucket_ratio,
        "bucket_ratio_config": dict(config.BUCKET_RATIO),
        "bucket_ratio_recommended": bucket_ratio_rec.get("recommended_bucket_ratio"),
        "bucket_ratio_decision": bucket_ratio_rec.get("decision"),
        "selected_model": str(selection_report.get("selected_model", "gp")),
        "gp_score": float(selection_report.get("gp_score", 0.0)) if selection_report.get("gp_score") is not None else None,
        "mlp_score": float(selection_report.get("mlp_score", 0.0)) if selection_report.get("mlp_score") is not None else None,
        "gp_composite_score": float(selection_report.get("gp_composite_score", 0.0)) if selection_report.get("gp_composite_score") is not None else None,
        "mlp_composite_score": float(selection_report.get("mlp_composite_score", 0.0)) if selection_report.get("mlp_composite_score") is not None else None,
        "holdout_accuracy": float(cm_metrics["accuracy"]) if cm_metrics else None,
        "holdout_precision": float(cm_metrics["precision"]) if cm_metrics else None,
        "holdout_recall": float(cm_metrics["recall"]) if cm_metrics else None,
        "holdout_f1": float(cm_metrics["f1"]) if cm_metrics else None,
        "holdout_tp": None,
        "holdout_fn": None,
        "holdout_tn": None,
        "holdout_fp": None,
        "holdout_tmax_rmse": float(tmax_metrics["rmse"]) if tmax_metrics else None,
        "holdout_tmax_r2": float(tmax_metrics["r2"]) if tmax_metrics else None,
        "holdout_tmax_n": int(tmax_metrics["n"]) if tmax_metrics else None,
        "boundary_count": int(bucket_counts_actual.get("boundary", 0)),
        "notp_high_tmax_count": int(bucket_counts_actual.get("notp_high_tmax", 0)),
        "uncertainty_sparse_count": int(bucket_counts_actual.get("uncertainty_sparse", 0)),
        "random_check_count": int(bucket_counts_actual.get("random_check", 0)),
        "fill_mixed_count": int(bucket_counts_actual.get("fill_mixed", 0)),
        "n_unique_combos_selected": n_unique_combos,
        "uncertainty_metric_source": uncertainty_source,
        "uncertainty_high_threshold": uncertainty_threshold,
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_p90": uncertainty_p90,
        "uncertainty_high_ratio": uncertainty_high_ratio,
    }

    # Extract confusion matrix counts if available
    if cm_metrics and final_test_csv and Path(final_test_csv).exists():
        from sklearn.metrics import confusion_matrix as sk_cm
        try:
            cm_arr = sk_cm(y_holdout_class, y_holdout_pred_class, labels=[config.NOTP_LABEL, config.TP_LABEL])
            iteration_summary["holdout_tn"] = int(cm_arr[0, 0])
            iteration_summary["holdout_fp"] = int(cm_arr[0, 1])
            iteration_summary["holdout_fn"] = int(cm_arr[1, 0])
            iteration_summary["holdout_tp"] = int(cm_arr[1, 1])
        except Exception:
            pass

    iteration_summary_path = try_dir / "iteration_summary.json"
    with open(iteration_summary_path, "w", encoding="utf-8") as f:
        json.dump(iteration_summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved iteration summary: {iteration_summary_path}")

    # === Collect history from Itr_n folders and include current run ===
    history_df = collect_iteration_history(config.OUTPUT_DIR)
    base_itr_num = int(history_df["itr_num"].max()) if len(history_df) > 0 and "itr_num" in history_df.columns else 0
    current_row = dict(iteration_summary)
    current_row["itr_num"] = base_itr_num + 1
    current_row["itr_folder"] = f"{try_dir.name}_current"
    current_row["is_current_run"] = True

    if len(history_df) > 0:
        history_df = history_df.copy()
        if "is_current_run" not in history_df.columns:
            history_df["is_current_run"] = False
        history_with_current = pd.concat([history_df, pd.DataFrame([current_row])], ignore_index=True)
    else:
        history_with_current = pd.DataFrame([current_row])

    history_with_current = history_with_current.sort_values("itr_num").reset_index(drop=True)

    trend_plot_path = performance_dir / "iteration_performance_trend.png"
    wrote_trend = save_iteration_performance_trend_plot(history_with_current, trend_plot_path)
    if wrote_trend:
        print(f"[INFO] Saved iteration performance trend plot (including current run): {trend_plot_path}")

    # Save history CSV
    history_csv_path = performance_dir / "iteration_history.csv"
    history_with_current.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved iteration history CSV (including current run): {history_csv_path}")

    # Recommend next batch size
    recommendation = recommend_next_batch_size(history_with_current, config)
    recommendation_path = try_dir / "batch_size_recommendation.json"
    with open(recommendation_path, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Batch size recommendation: {recommendation}")

    # Recommend next bucket ratio
    bucket_ratio_recommendation = recommend_bucket_ratio(history_with_current, config, effective_bucket_ratio)
    bucket_ratio_rec_path = try_dir / "bucket_ratio_recommendation.json"
    with open(bucket_ratio_rec_path, "w", encoding="utf-8") as f:
        json.dump(bucket_ratio_recommendation, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Bucket ratio recommendation: {bucket_ratio_recommendation}")

    written = [
        output_candidates_csv.name,
        output_scored_pool_csv.name,
        output_diagnostics_csv.name,
        output_cv_fold_metrics_csv.name,
        output_model_selection_json.name,
        output_optuna_report_json.name,
        output_pairs_dashboard_png.name,
        output_dashboard_png.name,
    ]
    if wrote_model_compare_png:
        written.append(output_model_compare_cv_png.name)
    (try_dir / "manifest.txt").write_text("\n".join(written), encoding="utf-8")
    print(f"[INFO] Saved all artifacts in: {try_dir}")
    print(f"[INFO] Artifact manifest: {try_dir / 'manifest.txt'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)