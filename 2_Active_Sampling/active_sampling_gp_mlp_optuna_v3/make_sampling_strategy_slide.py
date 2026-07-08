from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import config


def _fmt_weights(weights: dict[str, float]) -> list[str]:
    lines: list[str] = []
    for k, v in weights.items():
        lines.append(f"- {k}: {v:.2f}")
    return lines


def _block_text(title: str, ratio: float | None, lines: list[str], footer: list[str] | None = None) -> str:
    out = [title]
    if ratio is not None:
        out.append(f"ratio: {ratio:.2f}")
    out.append("")
    out.extend(lines)
    if footer:
        out.append("")
        out.extend(footer)
    return "\n".join(out)


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "#222222") -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        facecolor=fc,
        edgecolor=ec,
        alpha=0.97,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.012,
        y + h - 0.02,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="DejaVu Sans",
        color="#0f172a",
        linespacing=1.25,
    )


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color="#475569",
    )
    ax.add_patch(arr)


def build_slide(output_path: str = "outputs/sampling_strategy_one_slide.png") -> str:
    fig = plt.figure(figsize=(18, 10), dpi=180)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background gradient-like panels (simple layered rectangles)
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.0,rounding_size=0.02", facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1.0))
    ax.add_patch(FancyBboxPatch((0.02, 0.86), 0.96, 0.11, boxstyle="round,pad=0.008,rounding_size=0.02", facecolor="#e2e8f0", edgecolor="#cbd5e1", linewidth=1.0))

    title = "Active Sampling Strategy (Ver3) - One-Slide Overview"
    subtitle = "Buckets, scoring terms/weights, and active constraints from current config"
    ax.text(0.03, 0.935, title, fontsize=22, weight="bold", color="#0f172a", family="DejaVu Sans")
    ax.text(0.03, 0.895, subtitle, fontsize=12, color="#334155", family="DejaVu Sans")

    # Input/flow nodes
    input_text = "Candidate Pool\n- Continuous: A, C, E, F\n- Discrete: B, D\n- GP/MLP predictions\n  p_tp, tmax, uncertainty"
    _draw_box(ax, 0.03, 0.61, 0.22, 0.20, input_text, fc="#dbeafe")

    splitter_text = "Bucket Allocation\nby BUCKET_RATIO"
    _draw_box(ax, 0.29, 0.66, 0.14, 0.10, splitter_text, fc="#e9d5ff")

    _arrow(ax, 0.25, 0.71, 0.29, 0.71)

    # Bucket boxes
    boundary_lines_gp = _fmt_weights(config.BOUNDARY_WEIGHTS_GP)
    boundary_lines_mlp = _fmt_weights(config.BOUNDARY_WEIGHTS_MLP)
    boundary_footer = [
        "p_tp hard bound: 0.40~0.60",
        f"distance cols: {', '.join(config.BUCKET_LOCAL_DISTANCE_RULES['boundary']['cols'])}",
        f"distance min: {config.BUCKET_LOCAL_DISTANCE_RULES['boundary']['min_dist']:.2f}",
        "Cell_D quota: tp_ratio_only",
    ]
    boundary_text = _block_text(
        "Boundary Bucket",
        config.BUCKET_RATIO["boundary"],
        ["[GP weights]"] + boundary_lines_gp + ["", "[MLP weights]"] + boundary_lines_mlp,
        boundary_footer,
    )

    notp_lines = _fmt_weights(config.NOTP_HIGH_TMAX_WEIGHTS)
    notp_footer = [
        "p_tp hard bound: 0.10~0.50",
        f"distance cols: {', '.join(config.BUCKET_LOCAL_DISTANCE_RULES['notp_high_tmax']['cols'])}",
        f"distance min: {config.BUCKET_LOCAL_DISTANCE_RULES['notp_high_tmax']['min_dist']:.2f}",
        "Cell_D quota: tp_ratio_only",
    ]
    notp_text = _block_text(
        "NoTP High Tmax Bucket",
        config.BUCKET_RATIO["notp_high_tmax"],
        notp_lines,
        notp_footer,
    )

    us_lines = _fmt_weights(config.UNCERTAINTY_SPARSE_WEIGHTS)
    us_footer = [
        "focus: uncertainty + sparsity",
        "no explicit p_tp hard bound",
    ]
    us_text = _block_text(
        "Uncertainty Sparse Bucket",
        config.BUCKET_RATIO["uncertainty_sparse"],
        us_lines,
        us_footer,
    )

    rc_text = _block_text(
        "Random Check Bucket",
        config.BUCKET_RATIO["random_check"],
        ["- uniform random pick", "- exploration safety valve"],
        [f"global min distance: {config.MIN_BATCH_DISTANCE:.2f}"],
    )

    _draw_box(ax, 0.46, 0.52, 0.24, 0.34, boundary_text, fc="#fde68a")
    _draw_box(ax, 0.72, 0.52, 0.24, 0.34, notp_text, fc="#fecaca")
    _draw_box(ax, 0.46, 0.12, 0.24, 0.34, us_text, fc="#bbf7d0")
    _draw_box(ax, 0.72, 0.12, 0.24, 0.34, rc_text, fc="#bfdbfe")

    # arrows from splitter
    _arrow(ax, 0.43, 0.71, 0.46, 0.71)
    _arrow(ax, 0.43, 0.71, 0.72, 0.71)
    _arrow(ax, 0.36, 0.66, 0.52, 0.46)
    _arrow(ax, 0.36, 0.66, 0.78, 0.46)

    # Selection stage
    selection_text = (
        f"Greedy Selection + Diversity\n"
        f"- batch size: {config.BATCH_SIZE}\n"
        f"- per-bucket quotas by ratio\n"
        f"- global min distance: {config.MIN_BATCH_DISTANCE:.2f}\n"
        f"- output: next_sampling_candidates.csv"
    )
    _draw_box(ax, 0.03, 0.20, 0.38, 0.22, selection_text, fc="#ddd6fe")

    _arrow(ax, 0.58, 0.52, 0.30, 0.42)
    _arrow(ax, 0.84, 0.52, 0.30, 0.42)
    _arrow(ax, 0.58, 0.29, 0.30, 0.31)
    _arrow(ax, 0.84, 0.29, 0.30, 0.31)

    note = (
        "Ver3 key change: NOTP_HIGH_TMAX_WEIGHTS['notp_window']=0.00, tmax=0.70.\n"
        "This removed artificial p_tp=0.25 concentration and broadened notp_high_tmax selection."
    )
    ax.text(0.03, 0.06, note, fontsize=10.5, color="#334155", family="DejaVu Sans")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out = build_slide()
    print(f"Saved slide: {out}")
