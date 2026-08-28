from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

st = None

import config
import surrogate_bundle
from surrogate_bundle import get_bundle_info, load_surrogate_bundle, predict_with_bundle


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
    out_dir = Path(getattr(config, "OUTPUT_DIR", Path("outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "prediction_history.csv"


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
    return Path(getattr(config, "OUTPUT_DIR", Path("outputs"))) / "latest_surrogate_bundle.pkl"


def load_bundle_cached(bundle_path_str: str):
    return load_surrogate_bundle(bundle_path_str)


def show_bundle_panel(bundle):
    info = get_bundle_info(bundle)
    st.subheader("Model Version")
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


def single_predict_ui(bundle):
    info = get_bundle_info(bundle)
    bounds = info.get("continuous_bounds", {})
    levels = info.get("discrete_levels", {})
    base_continuous_cols = info.get("base_continuous_cols", [])
    discrete_cols = info.get("discrete_cols", [])

    st.subheader("Single Prediction")
    with st.form("single_predict_form"):
        c1, c2, c3 = st.columns(3)
        input_values = {}

        # Continuous inputs (split across first two columns), driven by the bundle.
        for i, col in enumerate(base_continuous_cols):
            lo, hi = bounds.get(col, (0.0, 1.0))
            target = c1 if i % 2 == 0 else c2
            input_values[col] = target.number_input(
                col,
                min_value=float(lo),
                max_value=float(hi),
                value=float((lo + hi) / 2),
                step=0.01,
            )

        # Discrete inputs, driven by the bundle.
        for col in discrete_cols:
            col_levels = levels.get(col, [])
            input_values[col] = c3.selectbox(col, options=col_levels, index=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_values])
        try:
            pred_df = predict_with_bundle(bundle, input_df)
            result_df = pd.concat([input_df, pred_df], axis=1)
            result_df.insert(0, "predicted_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            result_df.insert(1, "predict_mode", "single")
            append_history(result_df)
            st.success("Prediction succeeded")
            st.dataframe(result_df, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def batch_predict_ui(bundle):
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
            append_history(hist_df)
            st.success(f"Prediction succeeded: {len(out_df)} rows")
            st.dataframe(out_df.head(20), use_container_width=True)

            csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")


def history_ui():
    st.subheader("Prediction History")
    st.caption("예측 결과가 outputs/prediction_history.csv에 누적 저장됩니다.")

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

    st.write(f"Total rows: {len(hist)}")
    st.dataframe(hist.tail(200), use_container_width=True)
    st.download_button(
        "Download Full History CSV",
        data=hist.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )


def main():
    global st
    import streamlit as st_module
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        print("[ERROR] This app must be launched with Streamlit.")
        print("Run: streamlit run streamlit_surrogate_app.py")
        sys.exit(1)

    st = st_module

    st.set_page_config(page_title="Surrogate Predictor", layout="wide")
    st.title("Surrogate Predictor")
    st.caption("Saved surrogate bundle 기반 예측 도구")

    bundle_input = st.text_input("Bundle path", value=str(default_bundle_path()))
    bundle_path = Path(bundle_input)

    if not bundle_path.exists():
        st.error(f"Bundle not found: {bundle_path}")
        st.info("Run 0_main_active_sampling.py first to create outputs/latest_surrogate_bundle.pkl")
        st.stop()

    try:
        bundle = load_bundle_cached(str(bundle_path))
    except Exception as e:
        st.error(f"Failed to load bundle: {e}")
        st.stop()

    show_bundle_panel(bundle)
    st.divider()
    single_predict_ui(bundle)
    st.divider()
    batch_predict_ui(bundle)
    st.divider()
    history_ui()


if __name__ == "__main__":
    main()
