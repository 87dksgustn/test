import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path("outputs/final_test_doe/holdout_doe_50_from_initial_distribution.csv")
OUTPUT_DIR = Path("outputs/final_test_doe")
OUTPUT_PREFIX = "holdout_50"


def build_discrete_combo_id(df: pd.DataFrame) -> pd.Series:
    combo_cols = ["B_Barrier_Type", "D_Barrier_Outer_Type"]
    combos = df[combo_cols].astype(str).drop_duplicates().sort_values(combo_cols).reset_index(drop=True)
    mapping = {
        (row["B_Barrier_Type"], row["D_Barrier_Outer_Type"]): f"combo_{i + 1:03d}"
        for i, (_, row) in enumerate(combos.iterrows())
    }
    return df[combo_cols].astype(str).apply(lambda r: mapping[(r["B_Barrier_Type"], r["D_Barrier_Outer_Type"])], axis=1)


def save_combo_distribution(df: pd.DataFrame, out_path: Path, n_samples: int) -> None:
    combo_counts = (
        df.groupby(["discrete_combo_id", "D_Barrier_Outer_Type"]).size().unstack(fill_value=0).sort_index()
    )

    plt.figure(figsize=(14, 8), dpi=150)
    bottom = None
    colors = {"Si1": "#3A6FAA", "PU": "#74C09A"}
    for col in combo_counts.columns:
        vals = combo_counts[col]
        plt.bar(
            combo_counts.index,
            vals,
            bottom=bottom,
            label=str(col),
            color=colors.get(str(col), None),
        )
        bottom = vals if bottom is None else (bottom + vals)

    plt.title(f"Holdout DOE (n={n_samples}): Cases per Discrete Combo")
    plt.ylabel("Count")
    plt.xticks(rotation=60)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_continuous_distribution(df: pd.DataFrame, out_path: Path, n_samples: int) -> None:
    cols = ["A_Cell_D", "C_Barrier_Thx", "E_Barrier_Outer_Thx", "F_ThermalResin_Thx"]
    labels = sorted(df["D_Barrier_Outer_Type"].astype(str).unique().tolist())
    colors = ["#E76F51", "#2A9D8F"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    for ax, col in zip(axes.flatten(), cols):
        arrays = [df.loc[df["D_Barrier_Outer_Type"].astype(str) == lab, col].astype(float).to_numpy() for lab in labels]
        ax.hist(arrays, bins=20, stacked=True, label=labels, color=colors[: len(labels)], alpha=0.95)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"Holdout DOE (n={n_samples}): Continuous Distribution", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_scatter_cellD_vs_barrier(df: pd.DataFrame, out_path: Path, n_samples: int) -> None:
    labels = sorted(df["D_Barrier_Outer_Type"].astype(str).unique().tolist())
    colors = ["#E76F51", "#2A9D8F"]

    plt.figure(figsize=(14, 10), dpi=150)
    for i, lab in enumerate(labels):
        part = df.loc[df["D_Barrier_Outer_Type"].astype(str) == lab]
        plt.scatter(
            part["A_Cell_D"],
            part["C_Barrier_Thx"],
            s=110,
            c=colors[i % len(colors)],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            label=lab,
        )

    plt.title(f"Holdout DOE (n={n_samples}): Cell_D vs Barrier_Thx")
    plt.xlabel("A_Cell_D")
    plt.ylabel("C_Barrier_Thx")
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
    df["discrete_combo_id"] = build_discrete_combo_id(df)
    n_samples = len(df)

    combo_path = OUTPUT_DIR / f"{output_prefix}_combo_distribution.png"
    cont_path = OUTPUT_DIR / f"{output_prefix}_continuous_distribution.png"
    scatter_path = OUTPUT_DIR / f"{output_prefix}_cellD_vs_barrierThx.png"

    save_combo_distribution(df, combo_path, n_samples)
    save_continuous_distribution(df, cont_path, n_samples)
    save_scatter_cellD_vs_barrier(df, scatter_path, n_samples)

    print(f"Saved: {combo_path.as_posix()}")
    print(f"Saved: {cont_path.as_posix()}")
    print(f"Saved: {scatter_path.as_posix()}")


if __name__ == "__main__":
    main()