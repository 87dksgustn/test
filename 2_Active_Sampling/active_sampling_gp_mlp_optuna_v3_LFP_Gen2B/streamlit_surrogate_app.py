from __future__ import annotations

import itertools
import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna

st = None

import config
import surrogate_bundle
from surrogate_bundle import get_bundle_info, load_surrogate_bundle, predict_with_bundle


ADMIN_PASSWORD = "Q!w2p0o9"
CHEMISTRY_OPTIONS = ["LFP Gen2B", "LFP Gen2A", "HV Mid-Ni Gen1"]
INPUT_COLUMN_ORDER = [
    "A_Cell_D",
    "predicted_at",
    "B_Barrier_Type",
    "C_Barrier_Thx",
    "D_Barrier_Outer_Type",
    "E_Barrier_Outer_Thx",
    "F_ThermalResin_Thx",
    "G_CoolantLPM",
]
ADMIN_ONLY_RESULT_COLUMNS = {
    "predict_mode",
    "p_tp",
    "p_notp",
    "tmax_std",
    "clf_uncertainty",
    "MaxT_Adj_Y_pred",
    "MaxT_Adj_Y_std",
    "MaxT_Adj_Z_pred",
    "MaxT_Adj_Z_std",
    "Max_Power_std",
}
DEFAULT_OPT_TRIALS_PER_COMBO = 60
DEFAULT_OPT_TOP_N = 10
DEFAULT_OPT_PTP_UPPER = 0.50
DEFAULT_OPT_POPULATION_SIZE = 32


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def data_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return runtime_root()


def embedded_bundle_path() -> Path:
    return data_root() / "embedded_bundle" / "latest_surrogate_bundle.pkl"


def chemistry_bundle_candidates(chemistry: str):
    root = runtime_root()
    cwd = Path.cwd()
    mapping = {
        "LFP Gen2B": [
            embedded_bundle_path(),
            cwd / "outputs" / "latest_surrogate_bundle.pkl",
            root / "outputs" / "latest_surrogate_bundle.pkl",
        ],
        "LFP Gen2A": [
            cwd.parent / "active_sampling_gp_mlp_optuna_v3_LFP_Gen2A" / "outputs" / "latest_surrogate_bundle.pkl",
            root.parent / "active_sampling_gp_mlp_optuna_v3_LFP_Gen2A" / "outputs" / "latest_surrogate_bundle.pkl",
        ],
        "HV Mid-Ni Gen1": [
            cwd.parent / "active_sampling_gp_mlp_optuna_v3_MidNi_Gen1" / "outputs" / "latest_surrogate_bundle.pkl",
            root.parent / "active_sampling_gp_mlp_optuna_v3_MidNi_Gen1" / "outputs" / "latest_surrogate_bundle.pkl",
        ],
    }
    return [Path(p).resolve() for p in mapping.get(chemistry, [])]


def default_bundle_for_chemistry(chemistry: str) -> Path:
    candidates = chemistry_bundle_candidates(chemistry)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    return Path(getattr(config, "OUTPUT_DIR", Path("outputs"))) / "latest_surrogate_bundle.pkl"


def is_admin_mode() -> bool:
    return bool(st.session_state.get("is_admin", False))


def prediction_view_df(df: pd.DataFrame, admin_mode: bool) -> pd.DataFrame:
    ordered = [col for col in INPUT_COLUMN_ORDER if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    if not admin_mode:
        remaining = [col for col in remaining if col not in ADMIN_ONLY_RESULT_COLUMNS]
    return df.loc[:, ordered + remaining]


def display_table_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    return out


def render_centered_table(df: pd.DataFrame):
    table_html = display_table_df(df).to_html(index=False)
    st.markdown(
        """
<style>
.centered-table-wrap table {
  margin-left: auto;
  margin-right: auto;
  border-collapse: collapse;
}
.centered-table-wrap th,
.centered-table-wrap td {
  text-align: center !important;
}
</style>
"""
        + f"<div class='centered-table-wrap'>{table_html}</div>",
        unsafe_allow_html=True,
    )


def show_save_history_button(key: str, button_label: str):
    pending = pop_pending_history(key)
    if pending is None:
        return
    st.caption(f"Pending history rows: {len(pending)}")
    if st.button(button_label, key=f"save_{key}_btn"):
        append_history(pending)
        st.session_state.pop(key, None)
        st.success(f"Saved to history: {history_csv_path()}")


def mask_tp_regression_outputs(df: pd.DataFrame) -> pd.DataFrame:
    mask_fn = getattr(surrogate_bundle, "mask_tp_regression_outputs", None)
    if callable(mask_fn):
        return mask_fn(df)

    if "predicted_label" not in df.columns:
        return df

    out = df.copy()
    tp_mask = out["predicted_label"].astype(str).str.upper() == "TP"
    cols_to_mask = [col for col in ["tmax_pred", "tmax_std"] if col in out.columns]
    cols_to_mask.extend(
        col for col in out.columns
        if col.lower() in {"max_power_pred", "max_power_std"}
    )
    if cols_to_mask:
        out.loc[tp_mask, cols_to_mask] = pd.NA
    return out


def history_csv_path() -> Path:
    configured = Path(getattr(config, "OUTPUT_DIR", Path("outputs")))
    candidates = []

    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.append((Path.cwd() / configured).resolve())
        candidates.append((runtime_root() / configured).resolve())

    # Final fallback for read-only install locations.
    candidates.append((Path.home() / "SurrogatePredictor" / "outputs").resolve())

    checked = set()
    for out_dir in candidates:
        out_key = str(out_dir)
        if out_key in checked:
            continue
        checked.add(out_key)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            probe = out_dir / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return out_dir / "prediction_history.csv"
        except Exception:
            continue

    # Last-resort path if all checks fail.
    return (Path.cwd() / "prediction_history.csv").resolve()


def append_history(rows_df: pd.DataFrame):
    p = history_csv_path()
    rows_df = mask_tp_regression_outputs(rows_df)
    if p.exists():
        prev = pd.read_csv(p)
        merged = pd.concat([prev, rows_df], ignore_index=True)
    else:
        merged = rows_df.copy()
    merged = mask_tp_regression_outputs(merged)
    merged.to_csv(p, index=False, encoding="utf-8-sig")


def stash_pending_history(key: str, df: pd.DataFrame):
    st.session_state[key] = df.copy()


def pop_pending_history(key: str) -> pd.DataFrame | None:
    payload = st.session_state.get(key)
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    return None


def build_discrete_combinations(discrete_cols, discrete_levels):
    if not discrete_cols:
        return [{}]
    level_lists = []
    for col in discrete_cols:
        levels = list(discrete_levels.get(col, []))
        if not levels:
            levels = [""]
        level_lists.append(levels)
    combos = []
    for values in itertools.product(*level_lists):
        combos.append({col: val for col, val in zip(discrete_cols, values)})
    return combos


def sample_optimization_inputs(bundle, chemistry: str, trials_per_combo: int, seed: int, fixed_overrides=None):
    cfg = bundle.get("config", {})
    base_cont = list(cfg.get("base_continuous_cols", []))
    discrete_cols = list(cfg.get("discrete_cols", []))
    bounds = dict(cfg.get("continuous_bounds", {}))
    levels = dict(cfg.get("discrete_levels", {}))
    fixed_overrides = fixed_overrides or {}

    combos = build_discrete_combinations(discrete_cols, levels)
    for col in discrete_cols:
        if col in fixed_overrides:
            combos = [combo for combo in combos if str(combo.get(col)) == str(fixed_overrides[col])]

    rng = np.random.default_rng(seed)
    rows = []
    for combo_idx, combo in enumerate(combos, start=1):
        for _ in range(int(trials_per_combo)):
            row = {"combo_id": combo_idx}
            row.update(combo)
            for col in base_cont:
                if col in fixed_overrides:
                    row[col] = float(fixed_overrides[col])
                    continue
                lo, hi = bounds.get(col, (0.0, 1.0))
                lo_f = float(lo)
                hi_f = float(hi)
                row[col] = lo_f if lo_f == hi_f else float(rng.uniform(lo_f, hi_f))

            for col, val in fixed_overrides.items():
                row[col] = val

            if chemistry == "HV Mid-Ni Gen1":
                row.setdefault("D_Barrier_Outer_Type", "Si")
                row.setdefault("E_Barrier_Outer_Thx", 2.82)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _to_serializable(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def _trial_input_from_bundle(trial, cfg, chemistry: str, fixed_overrides):
    fixed_overrides = fixed_overrides or {}
    base_cont = list(cfg.get("base_continuous_cols", []))
    discrete_cols = list(cfg.get("discrete_cols", []))
    bounds = dict(cfg.get("continuous_bounds", {}))
    levels = dict(cfg.get("discrete_levels", {}))

    row = {}
    for col in base_cont:
        if col in fixed_overrides:
            row[col] = float(fixed_overrides[col])
            continue
        lo, hi = bounds.get(col, (0.0, 1.0))
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f == hi_f:
            row[col] = lo_f
        else:
            row[col] = float(trial.suggest_float(col, lo_f, hi_f))

    for col in discrete_cols:
        if col in fixed_overrides:
            row[col] = str(fixed_overrides[col])
            continue
        opts = list(levels.get(col, []))
        if not opts:
            opts = [""]
        row[col] = trial.suggest_categorical(col, opts)

    for col, val in fixed_overrides.items():
        row[col] = val

    if chemistry == "HV Mid-Ni Gen1":
        row.setdefault("D_Barrier_Outer_Type", "Si")
        row.setdefault("E_Barrier_Outer_Thx", 2.82)

    return row


def run_optuna_nsga2(bundle, chemistry: str, n_trials: int, seed: int, fixed_overrides, ptp_upper: float, population_size: int, progress=None, status=None):
    cfg = bundle.get("config", {})
    sampler = optuna.samplers.NSGAIISampler(
        seed=int(seed),
        population_size=max(2, int(population_size)),
    )
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=sampler,
    )

    def objective(trial):
        row = _trial_input_from_bundle(trial, cfg, chemistry, fixed_overrides)
        input_df = pd.DataFrame([row])
        pred_df = predict_with_bundle(bundle, input_df)
        out_row = pd.concat([input_df, pred_df], axis=1).iloc[0].to_dict()
        out_row = {k: _to_serializable(v) for k, v in out_row.items()}

        p_tp = float(out_row.get("p_tp", 0.0) or 0.0)
        tmax_given_notp = out_row.get("tmax_pred")
        tmax_given_notp = float(tmax_given_notp) if tmax_given_notp is not None else -1e9
        c_val = out_row.get("C_Barrier_Thx")
        c_val = float(c_val) if c_val is not None else 1e9

        # Keep constraint semantics while still allowing NSGA-II to explore.
        if p_tp >= float(ptp_upper):
            obj = (-1e6 + p_tp, -1e6 + tmax_given_notp, 1e6 + c_val)
        else:
            obj = (p_tp, tmax_given_notp, c_val)

        trial.set_user_attr("row", out_row)
        trial.set_user_attr("is_feasible", bool(p_tp < float(ptp_upper)))
        trial.set_user_attr("objective_1", float(obj[0]))
        trial.set_user_attr("objective_2", float(obj[1]))
        trial.set_user_attr("objective_3", float(obj[2]))
        return obj

    def callback(study_obj, trial):
        if progress is not None:
            done = trial.number + 1
            ratio = min(1.0, done / max(1, int(n_trials)))
            pct = int(25 + ratio * 35)
            progress.progress(pct, text=f"NSGA-II trials: {done}/{int(n_trials)}")
        if status is not None:
            status.info(f"Step 2/4: NSGA-II searching ({trial.number + 1}/{int(n_trials)})")

    study.optimize(objective, n_trials=int(n_trials), callbacks=[callback], show_progress_bar=False)

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = dict(trial.user_attrs.get("row", {}))
        row["trial_number"] = int(trial.number)
        row["optuna_value_1"] = float(trial.user_attrs.get("objective_1", 0.0))
        row["optuna_value_2"] = float(trial.user_attrs.get("objective_2", 0.0))
        row["optuna_value_3"] = float(trial.user_attrs.get("objective_3", 0.0))
        rows.append(row)

    return pd.DataFrame(rows)


def pareto_front_indices(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(values.shape[0], dtype=bool)
    for i in range(values.shape[0]):
        if not keep[i]:
            continue
        dominated_by_i = np.all(values[i] >= values, axis=1) & np.any(values[i] > values, axis=1)
        dominated_by_i[i] = False
        keep[dominated_by_i] = False
    return keep


def optimize_ui(bundle, admin_mode: bool, chemistry: str):
    st.subheader("Optimize")
    st.caption("Optuna NSGA-II optimization")

    c1, c2, c3 = st.columns(3)
    trials_per_combo = c1.number_input("Trials per combo", min_value=10, max_value=500, value=DEFAULT_OPT_TRIALS_PER_COMBO, step=10)
    top_n = c2.number_input("Top N", min_value=1, max_value=200, value=DEFAULT_OPT_TOP_N, step=1)
    seed = c3.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    if admin_mode:
        c4, c5 = st.columns(2)
        ptp_upper = c4.number_input("p_tp upper bound", min_value=0.0, max_value=1.0, value=DEFAULT_OPT_PTP_UPPER, step=0.01)
        population_size = c5.number_input("NSGA-II population size", min_value=2, max_value=512, value=DEFAULT_OPT_POPULATION_SIZE, step=2)
        st.caption("Experimental mode (Admin): threshold and population size are configurable.")
    else:
        ptp_upper = float(DEFAULT_OPT_PTP_UPPER)
        population_size = int(DEFAULT_OPT_POPULATION_SIZE)
        st.caption(f"Production mode: feasibility rule fixed at p_tp < {ptp_upper:.2f}")

    if int(population_size) > int(trials_per_combo):
        st.warning("Population size is larger than trials. Consider increasing trials for stable NSGA-II search.")

    cfg = bundle.get("config", {})
    base_cont = list(cfg.get("base_continuous_cols", []))
    discrete_cols = list(cfg.get("discrete_cols", []))
    bounds = dict(cfg.get("continuous_bounds", {}))
    levels = dict(cfg.get("discrete_levels", {}))
    fixable_vars = base_cont + discrete_cols

    selected_fixed_vars = st.multiselect(
        "Fixed variables",
        options=fixable_vars,
        default=[],
        help="Choose variables to keep fixed during optimization.",
    )

    fixed_overrides = {}
    if selected_fixed_vars:
        st.markdown("Fixed values")
    for col in selected_fixed_vars:
        if col in discrete_cols:
            opts = list(levels.get(col, []))
            if not opts:
                opts = [""]
            fixed_overrides[col] = st.selectbox(
                f"{col} fixed value",
                options=opts,
                index=0,
                key=f"opt_fix_{chemistry}_{col}",
            )
        else:
            lo, hi = bounds.get(col, (0.0, 1.0))
            lo_f = float(lo)
            hi_f = float(hi)
            fixed_overrides[col] = st.number_input(
                f"{col} fixed value",
                min_value=lo_f,
                max_value=hi_f,
                value=float((lo_f + hi_f) / 2),
                step=0.01,
                key=f"opt_fix_{chemistry}_{col}",
            )

    if fixed_overrides:
        st.caption("Applied fixed variables: " + ", ".join(f"{k}={v}" for k, v in fixed_overrides.items()))

    if not st.button("Run Optimization"):
        return

    progress = st.progress(0, text="Preparing optimization...")
    status = st.empty()

    status.info("Step 1/4: Setting search space")
    candidate_df = run_optuna_nsga2(
        bundle,
        chemistry,
        int(trials_per_combo),
        int(seed),
        fixed_overrides=fixed_overrides,
        ptp_upper=float(ptp_upper),
        population_size=int(population_size),
        progress=progress,
        status=status,
    )
    progress.progress(60, text="NSGA-II trials complete")
    if len(candidate_df) == 0:
        st.error("No completed Optuna trials were produced.")
        return

    out_df = candidate_df.copy()
    out_df.insert(0, "predicted_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    out_df.insert(1, "predict_mode", "optimize")
    out_df["tmax_pred_given_notp"] = out_df.get("tmax_pred")

    status.info("Step 3/4: Applying feasibility and ranking")
    feasible_df = out_df.loc[out_df["p_tp"] < float(ptp_upper)].copy() if "p_tp" in out_df.columns else out_df.copy()
    if len(feasible_df) == 0:
        progress.progress(100, text="Done")
        status.warning("Finished: no feasible candidates")
        st.warning("No feasible candidates found under the current p_tp threshold.")
        render_centered_table(prediction_view_df(out_df.head(20), admin_mode))
        stash_pending_history("pending_opt_history_json", out_df)
        show_save_history_button("pending_opt_history_json", "Save Last Optimization Result to History")
        return

    sort_cols = [col for col in ["p_tp", "tmax_pred_given_notp", "C_Barrier_Thx"] if col in feasible_df.columns]
    ascending = [False, False, True][:len(sort_cols)]
    ranked_df = feasible_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    top_df = ranked_df.head(int(top_n)).copy()
    top_df.insert(0, "opt_rank", np.arange(1, len(top_df) + 1, dtype=int))

    required_obj = ["p_tp", "tmax_pred_given_notp", "C_Barrier_Thx"]
    pareto_df = pd.DataFrame()
    if all(col in feasible_df.columns for col in required_obj):
        obj = np.column_stack([
            feasible_df["p_tp"].to_numpy(dtype=float),
            feasible_df["tmax_pred_given_notp"].to_numpy(dtype=float),
            -feasible_df["C_Barrier_Thx"].to_numpy(dtype=float),
        ])
        pareto_mask = pareto_front_indices(obj)
        pareto_df = feasible_df.loc[pareto_mask].copy()
        pareto_df = pareto_df.sort_values(["p_tp", "tmax_pred_given_notp", "C_Barrier_Thx"], ascending=[False, False, True]).reset_index(drop=True)
        pareto_df.insert(0, "pareto_rank", np.arange(1, len(pareto_df) + 1, dtype=int))

    progress.progress(100, text="Optimization complete")
    status.success("Step 4/4: Completed")
    st.success(f"Optimization done: total={len(out_df)}, feasible={len(feasible_df)}, top_n={len(top_df)}")

    st.markdown("Top Recommendations")
    render_centered_table(prediction_view_df(top_df, admin_mode))

    top_csv = prediction_view_df(top_df, admin_mode).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("Download Top N CSV", data=top_csv, file_name="opt_top_n.csv", mime="text/csv")

    st.markdown("Pareto Front")
    if len(pareto_df) == 0:
        st.info("Pareto front is unavailable for current output columns.")
    else:
        render_centered_table(prediction_view_df(pareto_df, admin_mode))
        pareto_csv = prediction_view_df(pareto_df, admin_mode).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("Download Pareto CSV", data=pareto_csv, file_name="opt_pareto.csv", mime="text/csv")

    show_plot = st.checkbox("Show 2D Pareto Plot", value=True)
    if show_plot:
        st.markdown("2D Pareto Plot")

        numeric_cols = [
            col for col in feasible_df.columns
            if pd.api.types.is_numeric_dtype(feasible_df[col])
        ]
        if len(numeric_cols) < 2:
            st.info("Not enough numeric columns to draw a 2D plot.")
        else:
            default_x = "C_Barrier_Thx" if "C_Barrier_Thx" in numeric_cols else numeric_cols[0]
            default_y = "tmax_pred_given_notp" if "tmax_pred_given_notp" in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])

            x_col = st.selectbox(
                "X axis",
                options=numeric_cols,
                index=numeric_cols.index(default_x) if default_x in numeric_cols else 0,
                key="opt_plot_x_axis",
            )
            y_col = st.selectbox(
                "Y axis",
                options=numeric_cols,
                index=numeric_cols.index(default_y) if default_y in numeric_cols else 0,
                key="opt_plot_y_axis",
            )

            color_candidates = ["p_tp", "tmax_std", "clf_uncertainty"]
            available_color_cols = [c for c in color_candidates if c in feasible_df.columns]
            if not available_color_cols:
                available_color_cols = ["none"]
            color_by = st.selectbox(
                "Color by",
                options=available_color_cols,
                index=0,
                key="opt_plot_color_by",
            )

            fig, ax = plt.subplots(figsize=(9, 6), dpi=140)

            if color_by == "none":
                ax.scatter(
                    feasible_df[x_col], feasible_df[y_col],
                    s=18, c="#b0b0b0", alpha=0.45, label="Feasible"
                )
            else:
                sc = ax.scatter(
                    feasible_df[x_col], feasible_df[y_col],
                    s=20, c=feasible_df[color_by], cmap="viridis", alpha=0.55, label="Feasible"
                )
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(color_by)

            if len(pareto_df) > 0 and x_col in pareto_df.columns and y_col in pareto_df.columns:
                pareto_sorted = pareto_df.sort_values(x_col, ascending=True)
                ax.plot(
                    pareto_sorted[x_col], pareto_sorted[y_col],
                    color="#1f77b4", linewidth=2.0, label="Pareto Front"
                )
                ax.scatter(
                    pareto_sorted[x_col], pareto_sorted[y_col],
                    s=36, c="#1f77b4", edgecolors="white", linewidths=0.6
                )

            if len(top_df) > 0 and x_col in top_df.columns and y_col in top_df.columns:
                ax.scatter(
                    top_df[x_col], top_df[y_col],
                    s=60, c="#d62728", marker="o", edgecolors="black", linewidths=0.7,
                    label="Top Recommendations"
                )
                if "opt_rank" in top_df.columns:
                    for _, row in top_df.iterrows():
                        ax.annotate(
                            str(int(row["opt_rank"])),
                            (row[x_col], row[y_col]),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=8,
                            color="#d62728",
                        )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Optimize Plot ({chemistry})")
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend(loc="best")
            fig.tight_layout()

            st.pyplot(fig, clear_figure=False)

            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
            png_buf.seek(0)
            st.download_button(
                "Download plot",
                data=png_buf.getvalue(),
                file_name="opt_pareto_2d_plot.png",
                mime="image/png",
            )
            plt.close(fig)

    stash_pending_history("pending_opt_history_json", out_df)
    show_save_history_button("pending_opt_history_json", "Save Last Optimization Result to History")


def load_history() -> pd.DataFrame:
    p = history_csv_path()
    if not p.exists():
        return pd.DataFrame()
    return mask_tp_regression_outputs(pd.read_csv(p))


def clear_history():
    p = history_csv_path()
    if p.exists():
        p.unlink()


def default_bundle_path() -> Path:
    embedded = embedded_bundle_path()
    if embedded.exists():
        return embedded
    return Path(getattr(config, "OUTPUT_DIR", Path("outputs"))) / "latest_surrogate_bundle.pkl"


def load_bundle_cached(bundle_path_str: str):
    return load_surrogate_bundle(bundle_path_str)


def show_bundle_panel(bundle):
    info = get_bundle_info(bundle)
    st.subheader("Bundle Details (Admin)")
    c1, c2, c3 = st.columns(3)
    c1.metric("model_id", str(info.get("model_id", "unknown")))
    c2.metric("created_at", str(info.get("created_at", "unknown")))
    c3.metric("selected_model", str(info.get("selected_model", "unknown")))

    st.write("- input_csv:", info.get("input_csv", ""))
    st.write("- try_dir:", info.get("try_dir", ""))
    st.write("- model_mode:", info.get("model_mode", ""))

    with st.expander("Input Schema / Bounds"):
        st.json(
            {
                "base_continuous_cols": info.get("base_continuous_cols", []),
                "discrete_cols": info.get("discrete_cols", []),
                "continuous_bounds": info.get("continuous_bounds", {}),
                "discrete_levels": info.get("discrete_levels", {}),
            }
        )


def range_help_text(lo: float, hi: float) -> str:
    return f"Range : [{float(lo):g}, {float(hi):g}]"


def number_input_with_hover_range(container, name: str, lo: float, hi: float):
    container.markdown(
        f"<span title=\"{range_help_text(lo, hi)}\"><b>{name}</b></span>",
        unsafe_allow_html=True,
    )
    return container.number_input(
        f"{name}_input",
        min_value=float(lo),
        max_value=float(hi),
        value=float((lo + hi) / 2),
        step=0.01,
        label_visibility="collapsed",
    )


def selectbox_with_label(container, name: str, options, index: int = 0):
    container.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
    return container.selectbox(
        f"{name}_input",
        options=options,
        index=index,
        label_visibility="collapsed",
    )


def single_predict_ui(bundle, admin_mode: bool, chemistry: str):
    info = get_bundle_info(bundle)
    bounds = info.get("continuous_bounds", {})
    levels = info.get("discrete_levels", {})

    st.subheader("Single Prediction")
    with st.form("single_predict_form"):
        c1, c2, c3 = st.columns(3)

        a_lo, a_hi = bounds.get("A_Cell_D", (8.0, 16.0))
        c_lo, c_hi = bounds.get("C_Barrier_Thx", (0.5, 2.5))
        e_lo, e_hi = bounds.get("E_Barrier_Outer_Thx", (1.1, 3.0))
        f_lo, f_hi = bounds.get("F_ThermalResin_Thx", (0.5, 2.5))
        g_lo, g_hi = bounds.get("G_CoolantLPM", (0.0, 30.0))

        b_levels = levels.get("B_Barrier_Type", ["Si1"])
        d_levels = levels.get("D_Barrier_Outer_Type", ["PU", "Si1"])

        A_Cell_D = number_input_with_hover_range(c1, "A_Cell_D", a_lo, a_hi)
        F_ThermalResin_Thx = number_input_with_hover_range(c1, "F_ThermalResin_Thx", f_lo, f_hi)
        G_CoolantLPM = None
        if chemistry == "HV Mid-Ni Gen1":
            G_CoolantLPM = number_input_with_hover_range(c1, "G_CoolantLPM", g_lo, g_hi)

        B_Barrier_Type = selectbox_with_label(c2, "B_Barrier_Type", options=b_levels, index=0)
        C_Barrier_Thx = number_input_with_hover_range(c2, "C_Barrier_Thx", c_lo, c_hi)

        if chemistry == "HV Mid-Ni Gen1":
            c3.markdown("<b>D_Barrier_Outer_Type</b>", unsafe_allow_html=True)
            D_Barrier_Outer_Type = c3.text_input(
                "D_Barrier_Outer_Type_input",
                value="Si",
                disabled=True,
                label_visibility="collapsed",
            )
            c3.markdown("<b>E_Barrier_Outer_Thx</b>", unsafe_allow_html=True)
            E_Barrier_Outer_Thx = c3.number_input(
                "E_Barrier_Outer_Thx_input_fixed",
                value=2.82,
                step=0.01,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            D_Barrier_Outer_Type = selectbox_with_label(c3, "D_Barrier_Outer_Type", options=d_levels, index=0)
            E_Barrier_Outer_Thx = number_input_with_hover_range(c3, "E_Barrier_Outer_Thx", e_lo, e_hi)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {
            "A_Cell_D": A_Cell_D,
            "C_Barrier_Thx": C_Barrier_Thx,
            "E_Barrier_Outer_Thx": E_Barrier_Outer_Thx,
            "F_ThermalResin_Thx": F_ThermalResin_Thx,
            "B_Barrier_Type": B_Barrier_Type,
            "D_Barrier_Outer_Type": D_Barrier_Outer_Type,
        }
        if chemistry == "HV Mid-Ni Gen1":
            row["G_CoolantLPM"] = G_CoolantLPM
        input_df = pd.DataFrame([row])
        try:
            pred_df = predict_with_bundle(bundle, input_df)
            result_df = pd.concat([input_df, pred_df], axis=1)
            result_df.insert(0, "predicted_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            result_df.insert(1, "predict_mode", "single")
            stash_pending_history("pending_single_history_json", result_df)
            st.success("Prediction succeeded")
            render_centered_table(prediction_view_df(result_df, admin_mode))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    show_save_history_button("pending_single_history_json", "Save This Single Prediction to History")


def batch_predict_ui(bundle, admin_mode: bool):
    st.subheader("Batch Prediction (CSV)")
    st.caption("필수 컬럼: A_Cell_D, C_Barrier_Thx, E_Barrier_Outer_Thx, F_ThermalResin_Thx, B_Barrier_Type, D_Barrier_Outer_Type")

    uploaded = st.file_uploader("Upload input CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        input_df = pd.read_csv(uploaded)
        st.write("Preview")
        st.dataframe(input_df.head(10), use_container_width=True)

        if st.button("Run Batch Prediction"):
            pred_df = predict_with_bundle(bundle, input_df)
            out_df = pd.concat([input_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            hist_df = out_df.copy()
            hist_df.insert(0, "predicted_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            hist_df.insert(1, "predict_mode", "batch")
            stash_pending_history("pending_batch_history_json", hist_df)
            st.success(f"Prediction succeeded: {len(out_df)} rows")
            shown_df = prediction_view_df(out_df, admin_mode)
            render_centered_table(shown_df.head(20))

            csv_bytes = shown_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )

        show_save_history_button("pending_batch_history_json", "Save Last Batch Prediction to History")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")


def history_ui(admin_mode: bool):
    st.subheader("Prediction History")
    st.caption(f"Current history file: {history_csv_path()}")
    st.caption("Prediction history is saved only when you press a Save button after prediction.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh History"):
            st.rerun()
    with col2:
        if st.button("Clear History"):
            clear_history()
            st.success("History cleared")
            st.rerun()

    hist = load_history()
    if len(hist) == 0:
        st.info("No accumulated prediction history yet.")
        return

    hist_view = prediction_view_df(hist, admin_mode)
    st.write(f"Total rows: {len(hist)}")
    render_centered_table(hist_view.tail(200))
    st.download_button(
        "Download Full History CSV",
        data=hist_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )


def main():
    global st
    import streamlit as st_module

    st = st_module

    if getattr(sys, "frozen", False):
        # In portable packaged mode, keep relative IO paths near the executable.
        try:
            os_root = runtime_root()
            os_root.mkdir(parents=True, exist_ok=True)
            import os
            os.chdir(os_root)
        except Exception:
            pass

    st.set_page_config(page_title="Surrogate Predictor", layout="wide")
    st.title("Surrogate Predictor")
    st.caption("Safety Simulation Team. All rights reserved")

    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False
    admin_mode = is_admin_mode()

    st.sidebar.header("Admin Mode")
    if admin_mode:
        st.sidebar.success("Admin mode enabled")
        if st.sidebar.button("Exit Admin Mode"):
            st.session_state["is_admin"] = False
            st.rerun()
    else:
        pw = st.sidebar.text_input("Admin password", type="password", key="admin_password_input")
        if st.sidebar.button("Enter Admin Mode"):
            if pw == ADMIN_PASSWORD:
                st.session_state["is_admin"] = True
                st.rerun()
            st.sidebar.error("Wrong password")

    st.subheader("Model")
    chemistry = st.selectbox("Chemistry", options=CHEMISTRY_OPTIONS, index=0)
    selected_default_bundle = default_bundle_for_chemistry(chemistry)

    if admin_mode:
        bundle_input = st.text_input("Bundle path", value=str(selected_default_bundle))
        bundle_path = Path(bundle_input)
    else:
        bundle_path = selected_default_bundle

    if not bundle_path.exists():
        st.error(f"Bundle not found: {bundle_path}")
        st.info("Create the selected chemistry surrogate bundle first, then retry.")
        st.stop()

    try:
        bundle = load_bundle_cached(str(bundle_path))
    except Exception as e:
        st.error(f"Failed to load bundle: {e}")
        st.stop()

    if admin_mode:
        show_bundle_panel(bundle)
        st.divider()

    st.caption(f"Active chemistry: {chemistry}")
    tabs = st.tabs(["Single Prediction", "Batch Prediction", "Prediction History", "Optimize"])
    with tabs[0]:
        single_predict_ui(bundle, admin_mode, chemistry)
    with tabs[1]:
        batch_predict_ui(bundle, admin_mode)
    with tabs[2]:
        history_ui(admin_mode)
    with tabs[3]:
        optimize_ui(bundle, admin_mode, chemistry)


if __name__ == "__main__":
    main()
