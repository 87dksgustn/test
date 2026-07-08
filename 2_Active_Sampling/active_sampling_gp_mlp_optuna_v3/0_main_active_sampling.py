import json
import itertools
import logging
import math
import re
from pathlib import Path

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
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
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
    fig.suptitle("Active Sampling 28-Point Selection Dashboard", fontsize=22, fontweight="bold", y=0.98)

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
    ax3.bar(idx, diag_plot["selected_new"], bottom=diag_plot["n_total"], label="Selected +28", color="#2A9D8F")
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
    output_pairs_dashboard_png = try_dir / "sampling_all_continuous_pairs_6panel.png"
    output_dashboard_png = try_dir / "sampling_selection_dashboard_1page.png"

    df = load_labeled_data(config.INPUT_CSV)
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
    print("[INFO] Model selection report:"); print(json.dumps(selection_report, indent=2, ensure_ascii=False))
    print(f"[INFO] Selected model kind: {getattr(selected_model, 'kind', 'gp')}")
    pool = generate_candidate_pool(valid_combos, config.CONTINUOUS_COLS, config.CONTINUOUS_BOUNDS, config.DISCRETE_COLS, config.CANDIDATES_PER_COMBO, config.EXCLUDED_REFERENCE_RANGES, config.RANDOM_SEED)
    print(f"[INFO] Candidate pool size after exclusion filter: {len(pool)}")
    x_candidate = pre.transform(pool[config.CONTINUOUS_COLS + config.DISCRETE_COLS].copy())
    scored = compute_acquisition_scores(pool, df, x_candidate, x_train, selected_model, config)
    scored.sort_values("acq_boundary", ascending=False).head(2000).to_csv(output_scored_pool_csv, index=False, encoding="utf-8-sig")
    bucket_target_counts = bucket_counts(config.BATCH_SIZE, config.BUCKET_RATIO)
    bucket_bin_quota_rules = build_bucket_bin_quota_rules(df, config, bucket_target_counts)

    selected = select_batch(
        scored,
        x_candidate,
        df,
        config.BATCH_SIZE,
        config.BUCKET_RATIO,
        config.MAX_SAMPLES_PER_COMBO,
        config.MIN_BATCH_DISTANCE,
        config.RANDOM_SEED,
        getattr(config, "BUCKET_DISTANCE_MULTIPLIER", {}),
        getattr(config, "BUCKET_LOCAL_DISTANCE_RULES", {}),
        bucket_bin_quota_rules,
        getattr(config, "BUCKET_PTP_BOUNDS", {}),
    )
    front = ["sampling_rank", "selected_bucket", "selected_model_kind"] + config.CONTINUOUS_COLS + config.DISCRETE_COLS + ["discrete_combo_id"]
    score_cols = ["p_tp", "p_notp", "boundary_score", "clf_uncertainty_raw", "clf_uncertainty_scaled", "tmax_pred_given_notp", "tmax_std_given_notp", "notp_window_score", "local_sparsity", "combo_priority", "acq_boundary", "acq_notp_high_tmax", "acq_uncertainty_sparse"]
    selected = selected[front + score_cols]
    selected.to_csv(output_candidates_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved next sampling candidates: {output_candidates_csv}")
    print(selected[["sampling_rank", "selected_bucket", "selected_model_kind", "discrete_combo_id", "p_tp", "p_notp", "tmax_pred_given_notp"]].head(20))

    save_continuous_pairs_dashboard(df, selected, config.CONTINUOUS_COLS, output_pairs_dashboard_png)
    print(f"[INFO] Saved continuous-pairs dashboard: {output_pairs_dashboard_png}")
    save_selection_dashboard(df, selected, diag, config, output_dashboard_png)
    print(f"[INFO] Saved dashboard: {output_dashboard_png}")

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
    (try_dir / "manifest.txt").write_text("\n".join(written), encoding="utf-8")
    print(f"[INFO] Saved all artifacts in: {try_dir}")
    print(f"[INFO] Artifact manifest: {try_dir / 'manifest.txt'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)