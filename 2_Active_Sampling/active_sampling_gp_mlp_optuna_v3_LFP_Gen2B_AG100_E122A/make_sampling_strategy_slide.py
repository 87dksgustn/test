from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import config


BG = "#f7f5ef"
NAVY = "#10253f"
TEXT = "#304255"
MUTED = "#6b7280"
CARD = "#fffdfa"
EDGE = "#d8d4ca"


def _box(ax, x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.0, r: float = 0.02) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )


def _text(ax, x: float, y: float, s: str, size: float, color: str = TEXT, weight: str = "normal", ha: str = "left", va: str = "top") -> None:
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, family="Malgun Gothic")


def _chip(ax, x: float, y: float, w: float, h: float, s: str, fc: str, color: str = "#ffffff", size: float = 9.8) -> None:
    _box(ax, x, y, w, h, fc=fc, ec=fc, lw=0.0, r=0.018)
    _text(ax, x + w / 2, y + h / 2, s, size, color=color, weight="bold", ha="center", va="center")


def _rows(ax, x: float, y_top: float, rows: list[tuple[str, float]], value_x: float, gap: float = 0.030, size: float = 9.5) -> float:
    for idx, (label, value) in enumerate(rows):
        y = y_top - idx * gap
        _text(ax, x, y, label, size)
        _text(ax, value_x, y, f"{value:.2f}", size, color=NAVY, weight="bold", ha="right")
    return y_top - len(rows) * gap


def _constraint_row(ax, x: float, y: float, labels: list[str], widths: list[float], colors: list[str]) -> None:
    cursor = x
    for label, width, color in zip(labels, widths, colors):
        _chip(ax, cursor, y, width, 0.032, label, fc=color, size=9.0)
        cursor += width + 0.010


def _card_shell(ax, x: float, y: float, w: float, h: float, title: str, ratio: float, accent: str, summary: str) -> None:
    _box(ax, x, y, w, h, fc=CARD, ec=EDGE, lw=1.0, r=0.025)
    ax.add_patch(Rectangle((x, y + h - 0.065), w, 0.065, facecolor=accent, edgecolor="none"))
    _text(ax, x + 0.018, y + h - 0.018, title, 16, color="#ffffff", weight="bold")
    _chip(ax, x + w - 0.10, y + h - 0.054, 0.08, 0.036, f"{int(ratio * 100)}%", fc=NAVY, size=10.0)
    _text(ax, x + 0.018, y + h - 0.082, summary, 10.5, color=MUTED)


def _boundary_card(ax, x: float, y: float, w: float, h: float) -> None:
    _card_shell(ax, x, y, w, h, "Boundary", config.BUCKET_RATIO["boundary"], "#295c9f", "p_tp = 0.5 부근의 분류 경계를 집중 공략합니다.")
    _constraint_row(
        ax,
        x + 0.018,
        y + h - 0.140,
        ["p_tp 0.40~0.60", "거리 0.10", "Cell_D quota"],
        [0.12, 0.10, 0.11],
        ["#1d4ed8", "#475569", "#7c2d12"],
    )
    _text(ax, x + 0.018, y + h - 0.170, "GP 가중치", 10.2, color=NAVY, weight="bold")
    _text(ax, x + 0.23, y + h - 0.170, "MLP 가중치", 10.2, color=NAVY, weight="bold")
    gp = list(config.BOUNDARY_WEIGHTS_GP.items())
    mlp = list(config.BOUNDARY_WEIGHTS_MLP.items())
    _rows(ax, x + 0.018, y + h - 0.200, gp, x + 0.18, gap=0.030, size=9.5)
    _rows(ax, x + 0.23, y + h - 0.200, mlp, x + w - 0.03, gap=0.030, size=9.5)


def _simple_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    ratio: float,
    accent: str,
    summary: str,
    rows: list[tuple[str, float]],
    constraints: list[str],
    constraint_widths: list[float],
    constraint_colors: list[str],
) -> None:
    _card_shell(ax, x, y, w, h, title, ratio, accent, summary)
    _constraint_row(ax, x + 0.018, y + h - 0.140, constraints, constraint_widths, constraint_colors)
    _text(ax, x + 0.018, y + h - 0.170, "Weights", 10.2, color=NAVY, weight="bold")
    _rows(ax, x + 0.018, y + h - 0.200, rows, x + w - 0.03, gap=0.030, size=9.5)


def build_slide(output_path: str = "outputs/sampling_strategy_one_slide.png") -> str:
    fig = plt.figure(figsize=(16, 9), dpi=180)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.add_patch(Rectangle((0, 0.885), 1, 0.115, facecolor="#c96a12", edgecolor="none"))
    _text(ax, 0.045, 0.952, "Active Sampling 전략", 28, color="#ffffff", weight="bold", va="center")
    _text(ax, 0.045, 0.913, "현재 버킷 구조, 점수 가중치, 하드 선택 규칙 요약", 12.8, color="#fff7ed", va="center")
    _chip(ax, 0.84, 0.912, 0.10, 0.05, "Ver3", fc="#1f3b5b", size=10.5)

    _box(ax, 0.045, 0.755, 0.91, 0.08, fc="#fff7ed", ec="#f2c68f", lw=1.0, r=0.025)
    _text(ax, 0.065, 0.804, "배치 크기 28 | boundary 60% | notp_high_tmax 30% | uncertainty_sparse 7% | random_check 3%", 13, color=NAVY, weight="bold")
    _text(ax, 0.065, 0.770, "점수화된 후보군에 대해 p_tp 하드 필터, Cell_D quota, 거리 기반 다양성 규칙을 적용해 greedy 선택을 수행합니다.", 11.0, color=MUTED)

    _chip(ax, 0.045, 0.700, 0.18, 0.038, f"전역 최소 거리 {config.MIN_BATCH_DISTANCE:.2f}", fc="#0f766e")
    _chip(ax, 0.235, 0.700, 0.24, 0.038, "거리 기준 축: A_Cell_D, C_Barrier_Thx", fc="#516072")
    _chip(ax, 0.485, 0.700, 0.20, 0.038, f"Cell_D quota: {config.BUCKET_CELLD_QUOTA_MODE}", fc="#7c4a1d")
    _chip(ax, 0.695, 0.700, 0.26, 0.038, "모델 출력: p_tp, p_notp, tmax, uncertainty", fc="#4a4fb3")

    _boundary_card(ax, 0.045, 0.385, 0.44, 0.28)
    _simple_card(
        ax,
        0.515,
        0.385,
        0.44,
        0.28,
        "NoTP High Tmax",
        config.BUCKET_RATIO["notp_high_tmax"],
        "#b94134",
        "NoTP 성향 후보 중 thermal risk가 큰 점을 우선 선택합니다.",
        list(config.NOTP_HIGH_TMAX_WEIGHTS.items()),
        ["p_tp 0.10~0.50", "notp_window 0.00", "Cell_D quota"],
        [0.12, 0.13, 0.11],
        ["#8f2d23", "#516072", "#7c4a1d"],
    )
    _simple_card(
        ax,
        0.045,
        0.085,
        0.44,
        0.25,
        "Uncertainty Sparse",
        config.BUCKET_RATIO["uncertainty_sparse"],
        "#2f8f61",
        "불확실하고 덜 커버된 영역에서 후보를 끌어옵니다.",
        list(config.UNCERTAINTY_SPARSE_WEIGHTS.items()),
        ["p_tp 하드바운드 없음", "diversity 중심"],
        [0.15, 0.13],
        ["#516072", "#226847"],
    )
    _simple_card(
        ax,
        0.515,
        0.085,
        0.44,
        0.25,
        "Random Check",
        config.BUCKET_RATIO["random_check"],
        "#6d4bc2",
        "블라인드 스팟 확인용 소규모 탐색 버킷입니다.",
        [("uniform_random_pick", 1.00)],
        ["모델 점수 비의존", "exploration safety valve"],
        [0.12, 0.16],
        ["#516072", "#6240b8"],
    )

    _text(ax, 0.045, 0.038, "Ver3 메모: notp_window = 0.00, tmax = 0.70으로 조정해 기존 p_tp ~= 0.25 집중 효과를 제거했습니다.", 11.2, color=MUTED)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    output = build_slide()
    print(f"Saved slide: {output}")
