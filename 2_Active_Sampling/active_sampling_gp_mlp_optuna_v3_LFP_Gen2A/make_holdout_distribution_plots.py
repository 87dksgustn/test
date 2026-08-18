import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Import config for column definitions
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


INPUT_CSV = Path("outputs/final_test_doe/holdout_doe_50_from_initial_distribution.csv")
OUTPUT_DIR = Path("outputs/final_test_doe")
OUTPUT_PREFIX = "holdout_50"


def build_discrete_combo_id(df: pd.DataFrame, discrete_cols: list) -> pd.Series:
    """Build combo IDs from discrete columns dynamically."""
    combos = df[discrete_cols].astype(str).drop_duplicates().sort_values(discrete_cols).reset_index(drop=True)
    mapping = {
        tuple(row[discrete_cols].astype(str).tolist()): f"combo_{i + 1:03d}"
        for i, (_, row) in enumerate(combos.iterrows())
    }
    return df[discrete_cols].astype(str).apply(lambda r: mapping[tuple(r.tolist())], axis=1)


def save_combo_distribution(df: pd.DataFrame, out_path: Path, n_samples: int, discrete_cols: list) -> None:
    # Use the last discrete column for color grouping (or first if only one)
    group_col = discrete_cols[-1] if discrete_cols else None
    if group_col and group_col in df.columns:
        combo_counts = (
            df.groupby(["discrete_combo_id", group_col]).size().unstack(fill_value=0).sort_index()
        )
    else:
        combo_counts = df.groupby("discrete_combo_id").size().to_frame(name="count").sort_index()

    plt.figure(figsize=(14, 8), dpi=150)
    bottom = None
    default_colors = ["#3A6FAA", "#74C09A", "#E76F51", "#9B59B6", "#F39C12"]
    for i, col in enumerate(combo_counts.columns):
        vals = combo_counts[col]
        plt.bar(
            combo_counts.index,
            vals,
            bottom=bottom,
            label=str(col),
            color=default_colors[i % len(default_colors)],
        )
        bottom = vals if bottom is None else (bottom + vals)

    plt.title(f"Holdout DOE (n={n_samples}): Cases per Discrete Combo")
    plt.ylabel("Count")
    plt.xticks(rotation=60)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_continuous_distribution(df: pd.DataFrame, out_path: Path, n_samples: int, continuous_cols: list, discrete_cols: list) -> None:
    cols = continuous_cols
    # Use the last discrete column for color grouping
    group_col = discrete_cols[-1] if discrete_cols else None
    if group_col and group_col in df.columns:
        labels = sorted(df[group_col].astype(str).unique().tolist())
    else:
        labels = ["all"]
    colors = ["#E76F51", "#2A9D8F", "#3A6FAA", "#9B59B6", "#F39C12"]

    n_cols = len(cols)
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows), dpi=150)
    axes_flat = axes.flatten() if n_cols > 1 else [axes]
    for i, col in enumerate(cols):
        ax = axes_flat[i]
        if group_col and group_col in df.columns:
            arrays = [df.loc[df[group_col].astype(str) == lab, col].astype(float).to_numpy() for lab in labels]
        else:
            arrays = [df[col].astype(float).to_numpy()]
        ax.hist(arrays, bins=20, stacked=True, label=labels, color=colors[: len(labels)], alpha=0.95)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    # Hide unused axes
    for j in range(n_cols, len(axes_flat)):
        axes_flat[j].set_visible(False)

    axes_flat[0].legend(loc="upper right")
    fig.suptitle(f"Holdout DOE (n={n_samples}): Continuous Distribution", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_scatter_cellD_vs_barrier(df: pd.DataFrame, out_path: Path, n_samples: int, continuous_cols: list, discrete_cols: list) -> None:
    # Use first two continuous columns for x/y
    x_col = continuous_cols[0] if len(continuous_cols) > 0 else None
    y_col = continuous_cols[1] if len(continuous_cols) > 1 else None
    if not x_col or not y_col:
        print("[WARN] Not enough continuous columns for scatter plot.")
        return

    group_col = discrete_cols[-1] if discrete_cols else None
    if group_col and group_col in df.columns:
        labels = sorted(df[group_col].astype(str).unique().tolist())
    else:
        labels = ["all"]
    colors = ["#E76F51", "#2A9D8F", "#3A6FAA", "#9B59B6", "#F39C12"]

    plt.figure(figsize=(14, 10), dpi=150)
    for i, lab in enumerate(labels):
        if group_col and group_col in df.columns:
            part = df.loc[df[group_col].astype(str) == lab]
        else:
            part = df
        plt.scatter(
            part[x_col],
            part[y_col],
            s=110,
            c=colors[i % len(colors)],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            label=lab,
        )

    plt.title(f"Holdout DOE (n={n_samples}): {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate holdout DOE distribution plots.")
    parser.add_argument("--input-csv", type=str, default=str(INPUT_CSV), help="Input holdout DOE CSV path.")
    parser.add_argument("--output-prefix", type=str, default=OUTPUT_PREFIX, help="Output PNG prefix.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_prefix = str(args.output_prefix)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    # Get column definitions from config
    discrete_cols = list(config.DISCRETE_COLS)
    continuous_cols = list(config.BASE_CONTINUOUS_COLS)

    df["discrete_combo_id"] = build_discrete_combo_id(df, discrete_cols)
    n_samples = len(df)

    combo_path = OUTPUT_DIR / f"{output_prefix}_combo_distribution.png"
    cont_path = OUTPUT_DIR / f"{output_prefix}_continuous_distribution.png"
    scatter_path = OUTPUT_DIR / f"{output_prefix}_scatter.png"

    save_combo_distribution(df, combo_path, n_samples, discrete_cols)
    save_continuous_distribution(df, cont_path, n_samples, continuous_cols, discrete_cols)
    save_scatter_cellD_vs_barrier(df, scatter_path, n_samples, continuous_cols, discrete_cols)

    print(f"Saved: {combo_path.as_posix()}")
    print(f"Saved: {cont_path.as_posix()}")
    print(f"Saved: {scatter_path.as_posix()}")


if __name__ == "__main__":
    main()