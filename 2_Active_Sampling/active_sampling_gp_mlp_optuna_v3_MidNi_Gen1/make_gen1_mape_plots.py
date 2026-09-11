from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_labeled_data
from surrogate_bundle import load_surrogate_bundle, predict_with_bundle


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "outputs" / "latest_surrogate_bundle.pkl"
FINAL_TEST_CSV = ROOT / "z_Final_Test_Dataset.csv"
OUT_DIR = ROOT / "outputs" / "final_test_doe"


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def target_to_pred_col(target_col: str, cfg: dict) -> str:
    if target_col == str(cfg.get("tmax_col", "")):
        return "tmax_pred"
    return f"{target_col}_pred"


def save_actual_vs_pred_plot(eval_df: pd.DataFrame, results_df: pd.DataFrame, out_png: Path) -> None:
    n_targets = len(results_df)
    n_cols = 3
    n_rows = int(math.ceil(n_targets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), dpi=150)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(n_rows, n_cols)

    for idx, row in results_df.reset_index(drop=True).iterrows():
        ax = axes[idx // n_cols, idx % n_cols]
        y_col = row["target"]
        p_col = row["pred_col"]

        yt = pd.to_numeric(eval_df[y_col], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(eval_df[p_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]

        ax.scatter(yt, yp, s=35, alpha=0.75, edgecolors="black", linewidths=0.3, c="#4C78A8")
        if len(yt) > 0:
            dmin = min(float(np.min(yt)), float(np.min(yp)))
            dmax = max(float(np.max(yt)), float(np.max(yp)))
            margin = max((dmax - dmin) * 0.05, 1e-6)
            lims = [dmin - margin, dmax + margin]
            ax.plot(lims, lims, "--", color="#E45756", linewidth=1.2)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        ax.set_title(f"{y_col}\nMAPE={row['mape_percent']:.2f}% (N={int(row['n'])})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.25)

    for j in range(n_targets, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle("Gen1 Final Test: Actual vs Predicted (NoTP only)", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_mape_bar_plot(results_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    x = np.arange(len(results_df))
    vals = results_df["mape_percent"].to_numpy(dtype=float)

    bars = ax.bar(x, vals, color="#2B6CB0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["target"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Gen1 Final Test MAPE by Output (NoTP only)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Surrogate bundle not found: {BUNDLE_PATH}")
    if not FINAL_TEST_CSV.exists():
        raise FileNotFoundError(f"Final test CSV not found: {FINAL_TEST_CSV}")

    bundle = load_surrogate_bundle(BUNDLE_PATH)
    cfg = bundle.get("config", {})

    data_df = load_labeled_data(FINAL_TEST_CSV)
    pred_df = predict_with_bundle(bundle, data_df)
    merged = pd.concat([data_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    pass_label = int(cfg.get("pass_label", 0))
    passfail_col = "TP_NoTP"
    if passfail_col not in merged.columns:
        raise ValueError(f"Required column not found: {passfail_col}")

    eval_df = merged.loc[pd.to_numeric(merged[passfail_col], errors="coerce") == pass_label].copy()
    if len(eval_df) == 0:
        raise ValueError("No NoTP rows found. MAPE evaluation requires at least one NoTP row.")

    target_cols = [str(cfg.get("tmax_col", ""))]
    target_cols.extend([str(c) for c in cfg.get("other_regression_cols", [])])
    target_cols.extend([str(c) for c in cfg.get("time_feature_cols", [])])
    target_cols = [c for c in target_cols if c and c in eval_df.columns]

    rows = []
    for col in target_cols:
        pred_col = target_to_pred_col(col, cfg)
        if pred_col not in eval_df.columns:
            continue
        yt = pd.to_numeric(eval_df[col], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(eval_df[pred_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) > 1e-12)
        rows.append(
            {
                "target": col,
                "pred_col": pred_col,
                "n": int(np.sum(mask)),
                "mape_percent": compute_mape(yt, yp),
            }
        )

    results_df = pd.DataFrame(rows)
    if len(results_df) == 0:
        raise ValueError("No target columns available for MAPE plotting.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_csv = OUT_DIR / "gen1_final_test_predictions_with_mape.csv"
    mape_csv = OUT_DIR / "gen1_final_test_mape_summary.csv"
    scatter_png = OUT_DIR / "gen1_actual_vs_pred_mape.png"
    bar_png = OUT_DIR / "gen1_mape_bar.png"

    merged.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    results_df.to_csv(mape_csv, index=False, encoding="utf-8-sig")
    save_actual_vs_pred_plot(eval_df, results_df, scatter_png)
    save_mape_bar_plot(results_df, bar_png)

    print(f"Saved predictions: {pred_csv}")
    print(f"Saved MAPE summary: {mape_csv}")
    print(f"Saved plot: {scatter_png}")
    print(f"Saved plot: {bar_png}")


if __name__ == "__main__":
    main()
