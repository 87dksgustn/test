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
CHEMISTRY_BUNDLE_TAG = {
    "LFP Gen2B": "LFP_Gen2B",
    "LFP Gen2A": "LFP_Gen2A",
    "HV Mid-Ni Gen1": "MidNi_Gen1",
}
CONFIDENCE_Z = {90: 1.645, 95: 1.960, 99: 2.576}
PTP_BOOTSTRAP_SAMPLES = 1500
PTP_BOOTSTRAP_N = 200
TOLERANCE_SAMPLES = 5000
TOLERANCE_LHS_SEED = 20260911

# Display-only placeholders for variables that are not part of a chemistry's
# surrogate schema yet. They are shown disabled and never sent to the model.
CHEMISTRY_PLACEHOLDER_INPUTS = {
    "HV Mid-Ni Gen1": {
        "D_Barrier_Outer_Type": "Si",
        "E_Barrier_Outer_Thx": 2.82,
    },
}

ADMIN_ONLY_RESULT_COLUMNS = {
    "predict_mode",
    "combo_id",
    "trial_number",
    "p_tp",
    "p_notp",
    "clf_uncertainty",
    "MaxT_Adj_Y_pred",
    "MaxT_Adj_Z_pred",
}


def is_admin_only_column(col: str) -> bool:
    name = str(col)
    lower = name.lower()
    if name in ADMIN_ONLY_RESULT_COLUMNS:
        return True
    if lower.startswith("maxt_adj_y") or lower.startswith("maxt_adj_z"):
        return True
    if name.endswith("_std"):
        return True
    if name.startswith("p_"):
        return True
    if "uncertainty" in lower:
        return True
    if name.startswith("optuna_value_"):
        return True
    return False


def bundle_schema(bundle) -> dict:
    cfg = bundle.get("config", {}) if isinstance(bundle, dict) else {}
    return {
        "base_continuous_cols": list(cfg.get("base_continuous_cols", [])),
        "discrete_cols": list(cfg.get("discrete_cols", [])),
        "continuous_bounds": dict(cfg.get("continuous_bounds", {})),
        "discrete_levels": dict(cfg.get("discrete_levels", {})),
        "other_regression_cols": list(cfg.get("other_regression_cols", [])),
        "time_feature_cols": list(cfg.get("time_feature_cols", [])),
    }


def schema_input_columns(bundle, chemistry: str) -> list[str]:
    schema = bundle_schema(bundle)
    cols = list(schema["base_continuous_cols"]) + list(schema["discrete_cols"])
    cols.extend(CHEMISTRY_PLACEHOLDER_INPUTS.get(chemistry, {}).keys())

    deduped = []
    seen = set()
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    return sorted(deduped)


def active_input_order() -> list[str]:
    order = st.session_state.get("active_input_order") if st is not None else None
    return list(order) if order else []


def active_chemistry() -> str:
    if st is None:
        return ""
    return str(st.session_state.get("active_chemistry", ""))


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


def embedded_bundle_paths_for_chemistry(chemistry: str) -> list[Path]:
    tag = CHEMISTRY_BUNDLE_TAG.get(chemistry, chemistry.replace(" ", "_"))
    return [
        # Preferred packaged layout: one folder per chemistry under embedded_bundle.
        data_root() / "embedded_bundle" / tag / "latest_surrogate_bundle.pkl",
        runtime_root() / "embedded_bundle" / tag / "latest_surrogate_bundle.pkl",
        # Legacy layout fallback: chemistry tag encoded in file name.
        data_root() / "embedded_bundle" / f"latest_surrogate_bundle_{tag}.pkl",
        runtime_root() / "embedded_bundle" / f"latest_surrogate_bundle_{tag}.pkl",
        # Accidental prior layout fallback (destination treated as directory).
        data_root() / "embedded_bundle" / f"latest_surrogate_bundle_{tag}.pkl" / "latest_surrogate_bundle.pkl",
        runtime_root() / "embedded_bundle" / f"latest_surrogate_bundle_{tag}.pkl" / "latest_surrogate_bundle.pkl",
        data_root() / "embedded_bundle" / "latest_surrogate_bundle.pkl",
        runtime_root() / "embedded_bundle" / "latest_surrogate_bundle.pkl",
    ]


def bundle_search_roots() -> list[Path]:
    roots = []
    for base in [runtime_root(), data_root(), Path.cwd()]:
        roots.extend([
            base,
            base / "outputs",
            base / "bundles",
            base / "embedded_bundle",
        ])
        # Sibling project folders (one per chemistry) next to this app.
        parent = base.parent
        if parent != base:
            roots.append(parent)
            try:
                roots.extend(sorted(p for p in parent.iterdir() if p.is_dir()))
            except Exception:
                pass
    roots.append(Path.home() / "SurrogatePredictor" / "outputs")

    deduped = []
    seen = set()
    for candidate in roots:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def chemistry_bundle_aliases(chemistry: str) -> list[str]:
    mapping = {
        "LFP Gen2B": [
            "LFP Gen2B",
            "LFP_Gen2B",
            "lfp_gen2b",
            "gen2b",
        ],
        "LFP Gen2A": [
            "LFP Gen2A",
            "LFP_Gen2A",
            "lfp_gen2a",
            "gen2a",
        ],
        "HV Mid-Ni Gen1": [
            "HV Mid-Ni Gen1",
            "HV_MidNi_Gen1",
            "MidNi_Gen1",
            "hv_midni_gen1",
            "midni_gen1",
        ],
    }
    aliases = mapping.get(chemistry, [chemistry])
    cleaned = []
    seen = set()
    for alias in aliases:
        normalized = alias.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    if not cleaned:
        cleaned = [chemistry]
    return cleaned


def _root_matches_alias(root: Path, aliases: list[str]) -> bool:
    name = root.name.lower().replace(" ", "_").replace("-", "")
    for alias in aliases:
        token = alias.lower().replace(" ", "_").replace("-", "")
        if token and token in name:
            return True
    return False


def chemistry_bundle_candidates(chemistry: str):
    aliases = chemistry_bundle_aliases(chemistry)
    other_aliases = []
    for other in CHEMISTRY_OPTIONS:
        if other != chemistry:
            other_aliases.extend(chemistry_bundle_aliases(other))

    named = []
    generic = []

    # Highest priority in packaged runtime: chemistry-specific embedded bundle.
    named.extend(embedded_bundle_paths_for_chemistry(chemistry))

    for root in bundle_search_roots():
        # A root belonging to a different chemistry must never be used.
        if _root_matches_alias(root, other_aliases) and not _root_matches_alias(root, aliases):
            continue

        for alias in aliases:
            named.extend([
                root / "outputs" / alias / "latest_surrogate_bundle.pkl",
                root / alias / "latest_surrogate_bundle.pkl",
                root / "embedded_bundle" / alias.replace(" ", "_") / "latest_surrogate_bundle.pkl",
                root / "embedded_bundle" / f"latest_surrogate_bundle_{alias.replace(' ', '_')}.pkl",
                root / "embedded_bundle" / f"latest_surrogate_bundle_{alias.replace(' ', '_')}.pkl" / "latest_surrogate_bundle.pkl",
                root / f"{alias}.pkl",
            ])

        if _root_matches_alias(root, aliases):
            named.extend([
                root / "outputs" / "latest_surrogate_bundle.pkl",
                root / "latest_surrogate_bundle.pkl",
            ])
        else:
            generic.extend([
                root / "outputs" / "latest_surrogate_bundle.pkl",
                root / "embedded_bundle" / "latest_surrogate_bundle.pkl",
            ])

    seen = set()
    ordered = []
    for candidate in named + generic:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(Path(resolved))
    return ordered


def default_bundle_for_chemistry(chemistry: str) -> Path:
    candidates = chemistry_bundle_candidates(chemistry)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    preferred = runtime_root() / "outputs" / chemistry_bundle_aliases(chemistry)[0] / "latest_surrogate_bundle.pkl"
    return preferred


def is_admin_mode() -> bool:
    return bool(st.session_state.get("is_admin", False))


def active_confidence_level() -> int:
    if st is None:
        return 95
    return int(st.session_state.get("confidence_level", 95))


def active_show_band() -> bool:
    if st is None:
        return True
    return bool(st.session_state.get("show_confidence_band", True))


def bootstrap_interval_for_ptp(values: pd.Series, level: int, n_size: int = PTP_BOOTSTRAP_N, n_boot: int = PTP_BOOTSTRAP_SAMPLES):
    alpha = (100.0 - float(level)) / 100.0
    lo_q = alpha / 2.0
    hi_q = 1.0 - lo_q
    rng = np.random.default_rng(20260910)

    lows = []
    highs = []
    for raw in values:
        p = pd.to_numeric(raw, errors="coerce")
        if pd.isna(p):
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        p = min(1.0, max(0.0, float(p)))
        draws = rng.binomial(n=int(n_size), p=p, size=int(n_boot)) / float(n_size)
        lows.append(float(np.quantile(draws, lo_q)))
        highs.append(float(np.quantile(draws, hi_q)))
    return pd.Series(lows, index=values.index), pd.Series(highs, index=values.index)


def add_confidence_band_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not active_show_band() or len(df) == 0:
        return df

    level = active_confidence_level()
    z = float(CONFIDENCE_Z.get(level, 1.960))
    out = df.copy()

    if "p_tp" in out.columns:
        lo_col = f"p_tp_ci{level}_lo"
        hi_col = f"p_tp_ci{level}_hi"
        band_col = f"p_tp_ci{level}_band"
        if lo_col not in out.columns or hi_col not in out.columns:
            lo_s, hi_s = bootstrap_interval_for_ptp(out["p_tp"], level=level)
            out[lo_col] = lo_s
            out[hi_col] = hi_s
        if band_col not in out.columns:
            out[band_col] = [
                "" if (pd.isna(lo) or pd.isna(hi)) else f"{float(lo):.2f}~{float(hi):.2f}"
                for lo, hi in zip(out[lo_col], out[hi_col])
            ]

    std_cols = [col for col in out.columns if str(col).endswith("_std")]
    for std_col in std_cols:
        base = str(std_col)[: -len("_std")]
        pred_candidates = [f"{base}_pred", base]
        pred_col = next((c for c in pred_candidates if c in out.columns), None)
        if pred_col is None:
            continue

        band_col = f"{base}_ci{level}_band"
        if band_col in out.columns:
            continue

        pred_num = pd.to_numeric(out[pred_col], errors="coerce")
        std_num = pd.to_numeric(out[std_col], errors="coerce")
        lower = pred_num - (z * std_num)
        upper = pred_num + (z * std_num)
        out[band_col] = [
            "" if (pd.isna(lo) or pd.isna(hi)) else f"{float(lo):.2f}~{float(hi):.2f}"
            for lo, hi in zip(lower, upper)
        ]

        # MidNi user view requires both labels: Time_Max_Power and Time_Tmax.
        if active_chemistry() == "HV Mid-Ni Gen1" and base == "Time_MaxT":
            out[f"Time_Tmax_ci{level}_band"] = out[band_col]

    return out


def prediction_view_df(df: pd.DataFrame, admin_mode: bool) -> pd.DataFrame:
    collapsed_mode = "__range_collapsed__" in df.columns and bool(df["__range_collapsed__"].astype(bool).any())
    if not collapsed_mode:
        df = add_confidence_band_columns(df)

    if "__range_collapsed__" in df.columns:
        df = df.drop(columns=["__range_collapsed__"], errors="ignore")

    level = active_confidence_level()
    ptp_lo_col = f"p_tp_ci{level}_lo"
    ptp_hi_col = f"p_tp_ci{level}_hi"

    tol_lo_col = "__p_tp_tol_lo"
    tol_hi_col = "__p_tp_tol_hi"
    if "predicted_label" in df.columns and collapsed_mode and tol_lo_col in df.columns and tol_hi_col in df.columns:
        def _label_with_tol_band(label, lo, hi):
            if pd.isna(lo) or pd.isna(hi):
                return label
            if float(lo) <= 0.5 <= float(hi):
                return f"{label}<br>(TP확률 : {float(lo):.2f}~{float(hi):.2f})"
            return label

        df = df.copy()
        df["predicted_label"] = [
            _label_with_tol_band(label, lo, hi)
            for label, lo, hi in zip(df["predicted_label"], df[tol_lo_col], df[tol_hi_col])
        ]

    if "predicted_label" in df.columns and ptp_lo_col in df.columns and ptp_hi_col in df.columns:
        def _label_with_ptp_band(label, lo, hi):
            if pd.isna(lo) or pd.isna(hi):
                return label
            if float(lo) <= 0.5 <= float(hi):
                return f"{label}<br>(TP확률 : {float(lo):.2f}~{float(hi):.2f})"
            return label

        df = df.copy()
        df["predicted_label"] = [
            _label_with_ptp_band(label, lo, hi)
            for label, lo, hi in zip(df["predicted_label"], df[ptp_lo_col], df[ptp_hi_col])
        ]

    if tol_lo_col in df.columns or tol_hi_col in df.columns:
        df = df.drop(columns=[tol_lo_col, tol_hi_col], errors="ignore")

    leading = ["predicted_at", "opt_rank", "pareto_rank"]
    ordered = [col for col in leading if col in df.columns]
    for col in active_input_order():
        if col in df.columns and col not in ordered:
            ordered.append(col)

    remaining = [col for col in df.columns if col not in ordered]
    if not admin_mode:
        remaining = [col for col in remaining if not is_admin_only_column(col)]
        filtered = []
        for col in remaining:
            if (not collapsed_mode) and str(col).endswith("_pred"):
                base = str(col)[: -len("_pred")]
                has_band = any(c.startswith(f"{base}_ci") and c.endswith("_band") for c in df.columns)
                if has_band:
                    continue
            filtered.append(col)
        remaining = filtered

    return df.loc[:, ordered + remaining]


def display_name_for_output(base_name: str) -> str:
    if base_name == "tmax":
        return "Tmax_adj"
    if base_name == "Time_MaxT":
        return "Time_Max_Power"
    return base_name


def display_table_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Format numeric values before renaming columns to avoid duplicate-name
    # collisions (e.g., pred and band both mapped to the same display label).
    numeric_cols = out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        name = str(col)
        if name in {"opt_rank", "pareto_rank", "trial_number", "combo_id"} or name.endswith("_rank"):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{int(float(x))}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")

    # Human-friendly headers for table display (schema independent).
    rename_map = {}
    for col in out.columns:
        name = str(col)
        if name == "predicted_label":
            rename_map[col] = "TP/No TP"
        elif "_ci" in name and name.endswith("_band"):
            head = name[: -len("_band")]
            ci_pos = head.find("_ci")
            target = head[:ci_pos]
            label_target = target[: -len("_pred")] if target.endswith("_pred") else target
            rename_map[col] = display_name_for_output(label_target)
        elif name.endswith("_pred"):
            rename_map[col] = display_name_for_output(name[: -len("_pred")])
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def render_centered_table(df: pd.DataFrame):
    table_html = display_table_df(df).to_html(index=False, escape=False)
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


def show_save_history_button(key: str, button_label: str, chemistry: str):
    pending = pop_pending_history(key)
    if pending is None:
        return
    st.caption(f"Pending history rows: {len(pending)}")
    if st.button(button_label, key=f"save_{key}_btn"):
        append_history(pending, chemistry)
        st.session_state.pop(key, None)
        st.success(f"Saved to history: {history_csv_path(chemistry)}")


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
        if col.lower() in {"max_power_pred", "max_power_std", "time_max_power_pred", "time_max_power_std"}
    )
    if cols_to_mask:
        out.loc[tp_mask, cols_to_mask] = pd.NA
    return out


def history_file_name(chemistry: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(chemistry)).strip("_")
    if not slug:
        slug = "default"
    return f"prediction_history_{slug}.csv"


def history_csv_path(chemistry: str) -> Path:
    configured = Path(getattr(config, "OUTPUT_DIR", Path("outputs")))
    candidates = []

    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.append((Path.cwd() / configured).resolve())
        candidates.append((runtime_root() / configured).resolve())

    # Final fallback for read-only install locations.
    candidates.append((Path.home() / "SurrogatePredictor" / "outputs").resolve())

    file_name = history_file_name(chemistry)
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
            return out_dir / file_name
        except Exception:
            continue

    # Last-resort path if all checks fail.
    return (Path.cwd() / file_name).resolve()


def append_history(rows_df: pd.DataFrame, chemistry: str):
    p = history_csv_path(chemistry)
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

            for col, val in CHEMISTRY_PLACEHOLDER_INPUTS.get(chemistry, {}).items():
                row.setdefault(col, val)
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

    for col, val in CHEMISTRY_PLACEHOLDER_INPUTS.get(chemistry, {}).items():
        row.setdefault(col, val)

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
        show_save_history_button("pending_opt_history_json", "Save Last Optimization Result to History", chemistry)
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
    show_save_history_button("pending_opt_history_json", "Save Last Optimization Result to History", chemistry)


def load_history(chemistry: str) -> pd.DataFrame:
    p = history_csv_path(chemistry)
    if not p.exists():
        return pd.DataFrame()
    return mask_tp_regression_outputs(pd.read_csv(p))


def clear_history(chemistry: str):
    p = history_csv_path(chemistry)
    if p.exists():
        p.unlink()


def default_bundle_path() -> Path:
    embedded = embedded_bundle_path()
    if embedded.exists():
        return embedded
    runtime_default = runtime_root() / "outputs" / "latest_surrogate_bundle.pkl"
    if runtime_default.exists():
        return runtime_default
    data_default = data_root() / "outputs" / "latest_surrogate_bundle.pkl"
    if data_default.exists():
        return data_default
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


def number_input_with_hover_range(container, name: str, lo: float, hi: float, key: str | None = None):
    container.markdown(
        f"<span title=\"{range_help_text(lo, hi)}\"><b>{name}</b></span>",
        unsafe_allow_html=True,
    )
    lo_f = float(lo)
    hi_f = float(hi)
    return container.number_input(
        f"{name}_input",
        min_value=lo_f,
        max_value=hi_f,
        value=float((lo_f + hi_f) / 2),
        step=0.01,
        label_visibility="collapsed",
        key=key,
    )


def selectbox_with_label(container, name: str, options, index: int = 0, key: str | None = None):
    container.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
    return container.selectbox(
        f"{name}_input",
        options=options,
        index=index,
        label_visibility="collapsed",
        key=key,
    )


def disabled_placeholder_input(container, name: str, value, key: str | None = None):
    container.markdown(
        f"<span title=\"Not used by the current model\"><b>{name}</b></span>",
        unsafe_allow_html=True,
    )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        container.number_input(
            f"{name}_input_fixed",
            value=float(value),
            step=0.01,
            disabled=True,
            label_visibility="collapsed",
            key=key,
        )
    else:
        container.text_input(
            f"{name}_input_fixed",
            value=str(value),
            disabled=True,
            label_visibility="collapsed",
            key=key,
        )


def chunk_into_columns(items: list[str], n_cols: int) -> list[list[str]]:
    groups: list[list[str]] = [[] for _ in range(n_cols)]
    if not items:
        return groups
    per = -(-len(items) // n_cols)
    for idx, item in enumerate(items):
        groups[min(idx // per, n_cols - 1)].append(item)
    return groups


def build_single_input_layout(ordered_vars: list[str]) -> list[list[str]]:
    # Keep the Gen2 familiar grouping: B/C pair and D/E pair vertically aligned.
    col1, col2, col3 = [], [], []
    used = set()

    pair_specs = [
        ("B_Barrier_Type", "C_Barrier_Thx", col2),
        ("D_Barrier_Outer_Type", "E_Barrier_Outer_Thx", col3),
    ]
    vars_set = set(ordered_vars)
    for type_col, thx_col, target_col in pair_specs:
        if type_col in vars_set:
            target_col.append(type_col)
            used.add(type_col)
        if thx_col in vars_set:
            target_col.append(thx_col)
            used.add(thx_col)

    for var in ordered_vars:
        if var in used:
            continue
        col1.append(var)

    if not col1 and ordered_vars:
        # Fallback for uncommon schemas where all variables are consumed by pairs.
        col1 = [ordered_vars[0]]
    return [col1, col2, col3]


def annotate_inputs_with_tolerance(df: pd.DataFrame, tolerances: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for var, tol in tolerances.items():
        if var not in out.columns:
            continue
        out[var] = [
            "" if pd.isna(v) else f"{float(v):.2f}<br>(+/-{float(tol):.1f}%)"
            for v in out[var]
        ]
    return out


def lhs_tolerance_rows(
    base_row: dict,
    tolerances: dict[str, float],
    bounds: dict,
    n_samples: int,
) -> tuple[list[dict], list[str]]:
    from scipy.stats import qmc

    tol_vars = sorted(tolerances.keys())
    lows = []
    highs = []
    clipped_vars = []

    for var in tol_vars:
        base_val = float(base_row[var])
        tol = float(tolerances[var]) / 100.0
        lo_raw = base_val * (1.0 - tol)
        hi_raw = base_val * (1.0 + tol)
        model_lo, model_hi = bounds.get(var, (lo_raw, hi_raw))
        lo = max(lo_raw, float(model_lo))
        hi = min(hi_raw, float(model_hi))
        if lo > hi:
            lo = hi = min(max(base_val, float(model_lo)), float(model_hi))
        if lo != lo_raw or hi != hi_raw:
            clipped_vars.append(var)
        lows.append(lo)
        highs.append(hi)

    sampler = qmc.LatinHypercube(d=len(tol_vars), seed=TOLERANCE_LHS_SEED)
    unit = sampler.random(n=int(n_samples))
    scaled = qmc.scale(unit, lows, highs)

    rows = []
    for sample in scaled:
        sampled_row = dict(base_row)
        for idx, var in enumerate(tol_vars):
            sampled_row[var] = float(sample[idx])
        rows.append(sampled_row)
    return rows, clipped_vars


def collapse_tolerance_scenarios_to_range_row(df: pd.DataFrame, input_cols: list[str]) -> pd.DataFrame:
    if len(df) <= 1 or "percent_case" not in df.columns:
        return df

    base_mask = df["percent_case"].astype(str).str.lower().eq("base")
    base_row = df.loc[base_mask].head(1)
    if base_row.empty:
        base_row = df.head(1)

    collapsed = dict(base_row.iloc[0].to_dict())
    collapsed.pop("percent_case", None)

    keep_as_is = set(input_cols) | {"predicted_at", "predict_mode", "opt_rank", "pareto_rank"}

    for col in df.columns:
        if col in keep_as_is or col == "percent_case":
            continue

        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            lo = float(numeric.min())
            hi = float(numeric.max())
            collapsed[col] = f"{lo:.2f}~{hi:.2f}"
            continue

        values = []
        for v in series:
            if pd.isna(v):
                continue
            text = str(v)
            if text == "":
                continue
            if text not in values:
                values.append(text)

        if not values:
            collapsed[col] = ""
        elif len(values) == 1:
            collapsed[col] = values[0]
        else:
            collapsed[col] = " / ".join(values)

    # TP/NoTP is always reported from the base prediction result.
    if "predicted_label" in base_row.columns:
        collapsed["predicted_label"] = str(base_row.iloc[0]["predicted_label"])

    # Keep tolerance-induced TP probability range as hidden numeric bounds
    # so we can annotate predicted_label when it crosses 0.5.
    if "p_tp" in df.columns:
        p_num = pd.to_numeric(df["p_tp"], errors="coerce")
        if p_num.notna().any():
            collapsed["__p_tp_tol_lo"] = float(p_num.min())
            collapsed["__p_tp_tol_hi"] = float(p_num.max())

    collapsed["__range_collapsed__"] = True

    return pd.DataFrame([collapsed])


def single_predict_ui(bundle, admin_mode: bool, chemistry: str):
    schema = bundle_schema(bundle)
    bounds = schema["continuous_bounds"]
    levels = schema["discrete_levels"]
    continuous_cols = schema["base_continuous_cols"]
    discrete_cols = schema["discrete_cols"]
    placeholders = CHEMISTRY_PLACEHOLDER_INPUTS.get(chemistry, {})

    ordered_vars = schema_input_columns(bundle, chemistry)

    st.subheader("Single Prediction")
    st.caption(f"Model inputs for {chemistry}: " + ", ".join(sorted(continuous_cols + discrete_cols)))

    # Keep these controls outside the form so they react immediately without pressing Predict.
    mode_cols = st.columns([1.0, 3.0])
    mode_cols[0].markdown(
        """
<style>
.tol-label-wrap {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    position: relative;
}
.tol-help-icon {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 1px solid rgba(120, 120, 120, 0.55);
    color: rgba(110, 110, 110, 0.9);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    background: rgba(220, 220, 220, 0.25);
    cursor: help;
    position: relative;
}
.tol-help-icon::after {
    content: "Set a tolerance per continuous variable. The tool explores the combined range with Latin Hypercube sampling and reports min~max of the predicted outputs.";
    position: absolute;
    left: 22px;
    top: 50%;
    transform: translateY(-50%);
    min-width: 280px;
    max-width: 360px;
    background: #2b2b2b;
    color: #f2f2f2;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.35;
    border-radius: 6px;
    padding: 8px 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    z-index: 50;
    white-space: normal;
}
.tol-help-icon:hover::after {
    opacity: 1;
    visibility: visible;
}
</style>
<span class="tol-label-wrap"><b>Tolerance</b><span class="tol-help-icon">!</span></span>
""",
        unsafe_allow_html=True,
    )
    use_percent_for_cont = mode_cols[0].toggle(
        "tolerance_toggle",
        value=False,
        label_visibility="collapsed",
        key=f"single_percent_mode_{chemistry}",
    )

    var_tolerances: dict[str, float] = {}
    if use_percent_for_cont and continuous_cols:
        with st.expander("Per-variable tolerance (%)", expanded=True):
            tol_cols = st.columns(min(3, len(continuous_cols)))
            for idx, var in enumerate(sorted(continuous_cols)):
                target = tol_cols[idx % len(tol_cols)]
                var_tolerances[var] = target.slider(
                    var,
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.1,
                    key=f"single_tol_{chemistry}_{var}",
                )

    with st.form(f"single_predict_form_{chemistry}"):
        cols = st.columns(3)
        groups = build_single_input_layout(ordered_vars)

        row = {}
        for col_idx, group in enumerate(groups[:3]):
            container = cols[col_idx]
            for var in group:
                widget_key = f"single_{chemistry}_{var}"
                if var in placeholders:
                    disabled_placeholder_input(container, var, placeholders[var], key=widget_key)
                elif var in discrete_cols:
                    opts = list(levels.get(var, []))
                    if not opts:
                        opts = [""]
                    row[var] = selectbox_with_label(container, var, options=opts, index=0, key=widget_key)
                else:
                    lo, hi = bounds.get(var, (0.0, 1.0))
                    row[var] = number_input_with_hover_range(container, var, lo, hi, key=f"{widget_key}_abs")

        submitted = st.form_submit_button("Predict")

    if submitted:
        scenario_rows = []
        active_tolerances = {
            var: float(tol)
            for var, tol in var_tolerances.items()
            if float(tol) > 0.0 and var in row
        }

        if use_percent_for_cont and active_tolerances:
            base_row = dict(row)
            base_row["percent_case"] = "base"

            sampled_rows, clipped_vars = lhs_tolerance_rows(
                row,
                active_tolerances,
                bounds,
                TOLERANCE_SAMPLES,
            )
            for sampled in sampled_rows:
                sampled["percent_case"] = "sample"

            scenario_rows = [base_row] + sampled_rows

            st.info(
                f"Explored {len(sampled_rows)} LHS samples across "
                f"{len(active_tolerances)} continuous variables."
            )
            if clipped_vars:
                st.warning(
                    "Some sampled values exceeded model bounds and were clipped: "
                    + ", ".join(clipped_vars)
                )
        else:
            scenario_rows = [dict(row)]

        input_df = pd.DataFrame(scenario_rows)
        try:
            pred_df = predict_with_bundle(bundle, input_df)
            result_df = pd.concat([input_df, pred_df], axis=1)
            result_df.insert(0, "predicted_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            result_df.insert(1, "predict_mode", "single")
            result_df = collapse_tolerance_scenarios_to_range_row(result_df, ordered_vars)
            stash_pending_history("pending_single_history_json", result_df)
            st.success("Prediction succeeded")
            shown_df = prediction_view_df(result_df, admin_mode)
            if active_tolerances:
                shown_df = annotate_inputs_with_tolerance(shown_df, active_tolerances)
            render_centered_table(shown_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    show_save_history_button("pending_single_history_json", "Save This Single Prediction to History", chemistry)


def batch_predict_ui(bundle, admin_mode: bool, chemistry: str):
    schema = bundle_schema(bundle)
    required_cols = list(schema["base_continuous_cols"]) + list(schema["discrete_cols"])

    st.subheader("Batch Prediction (CSV)")
    st.caption(f"{chemistry} 필수 컬럼: " + ", ".join(sorted(required_cols)))

    uploaded = st.file_uploader("Upload input CSV", type=["csv"], key=f"batch_upload_{chemistry}")
    if uploaded is None:
        return

    try:
        input_df = pd.read_csv(uploaded)
        missing = [col for col in required_cols if col not in input_df.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            return

        st.write("Preview")
        st.dataframe(input_df.head(10), use_container_width=True)

        if st.button("Run Batch Prediction", key=f"batch_run_{chemistry}"):
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
                file_name=f"predictions_{history_file_name(chemistry)}",
                mime="text/csv",
            )

        show_save_history_button("pending_batch_history_json", "Save Last Batch Prediction to History", chemistry)
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")


def history_ui(admin_mode: bool, chemistry: str):
    st.subheader("Prediction History")
    st.caption(f"Chemistry: {chemistry}")
    st.caption(f"Current history file: {history_csv_path(chemistry)}")
    st.caption("Prediction history is saved only when you press a Save button after prediction.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh History", key=f"hist_refresh_{chemistry}"):
            st.rerun()
    with col2:
        if st.button("Clear History", key=f"hist_clear_{chemistry}"):
            clear_history(chemistry)
            st.success("History cleared")
            st.rerun()

    hist = load_history(chemistry)
    if len(hist) == 0:
        st.info("No accumulated prediction history yet.")
        return

    hist = hist.dropna(axis=1, how="all")
    hist_view = prediction_view_df(hist, admin_mode)
    st.write(f"Total rows: {len(hist)}")
    render_centered_table(hist_view.tail(200))
    st.download_button(
        "Download Full History CSV",
        data=hist_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name=history_file_name(chemistry),
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
    model_col1, model_col2, model_col3 = st.columns([1.2, 1.0, 1.0])
    chemistry = model_col1.selectbox("Chemistry", options=CHEMISTRY_OPTIONS, index=0)
    model_col2.selectbox(
        "Confidence Level",
        options=[90, 95, 99],
        index=1,
        key="confidence_level",
        help="Prediction band is computed as pred ± z * std.",
    )
    model_col3.checkbox(
        "Show Band",
        value=True,
        key="show_confidence_band",
        help="When enabled, lower/upper confidence band columns are added to results.",
    )
    selected_default_bundle = default_bundle_for_chemistry(chemistry)

    if admin_mode:
        # Prevent stale admin path when chemistry selection changes.
        if st.session_state.get("bundle_path_for_chem") != chemistry:
            st.session_state["bundle_path_for_chem"] = chemistry
            st.session_state["bundle_path_input"] = str(selected_default_bundle)
        bundle_input = st.text_input("Bundle path", key="bundle_path_input")
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

    st.session_state["active_input_order"] = schema_input_columns(bundle, chemistry)
    st.session_state["active_chemistry"] = chemistry

    st.caption(f"Active chemistry: {chemistry}")
    tabs = st.tabs(["Single Prediction", "Batch Prediction", "Prediction History", "Optimize"])
    with tabs[0]:
        single_predict_ui(bundle, admin_mode, chemistry)
    with tabs[1]:
        batch_predict_ui(bundle, admin_mode, chemistry)
    with tabs[2]:
        history_ui(admin_mode, chemistry)
    with tabs[3]:
        optimize_ui(bundle, admin_mode, chemistry)


if __name__ == "__main__":
    main()
