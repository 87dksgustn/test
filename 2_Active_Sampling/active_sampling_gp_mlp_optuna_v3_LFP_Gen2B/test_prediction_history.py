import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import streamlit_surrogate_app as app


class PredictionHistoryTests(unittest.TestCase):
    def setUp(self):
        streamlit_patch = patch.object(app, "st", SimpleNamespace(session_state={}))
        streamlit_patch.start()
        self.addCleanup(streamlit_patch.stop)

    def configure(self, chemistry, mode):
        app.st.session_state.update({
            "active_chemistry": chemistry,
            "active_input_order": ["A_Cell_D"],
            "admin_show_confidence_band_persist": mode == "band",
            "admin_confidence_level_persist": 95,
        })

    def predictions(self, chemistry, mode):
        raw = pd.DataFrame({
            "predicted_at": ["2026-10-01 10:00:00", "2026-10-01 10:01:00"],
            "A_Cell_D": [12.4, 13.375],
            "predicted_label": ["TP", "NoTP"],
            "p_tp": [0.9, 0.1],
            "p_notp": [0.1, 0.9],
            "clf_uncertainty": [0.2, 0.2],
            "predict_mode": ["single", "single"],
        })
        for base in app.chemistry_output_base_order(chemistry)[1:]:
            raw[f"{base}_pred"] = ["10.00~20.00", "30.00~40.00"] if mode == "tolerance" else [10.123, 30.123]
            raw[f"{base}_std"] = [0.5, 0.5]
        if mode == "tolerance":
            raw["__range_collapsed__"] = True
            raw["__p_tp_tol_lo"] = [0.4, 0.3]
            raw["__p_tp_tol_hi"] = [0.7, 0.6]
        return raw

    def single_view(self, chemistry, mode):
        view = app.prediction_view_df(self.predictions(chemistry, mode), False)
        if mode == "tolerance":
            view = app.annotate_inputs_with_tolerance(view, {"A_Cell_D": 5.0})
        return view

    def test_display_and_saved_values_match_for_all_models(self):
        for chemistry in app.CHEMISTRY_OPTIONS:
            for mode in ["point", "band", "tolerance"]:
                with self.subTest(chemistry=chemistry, mode=mode):
                    self.configure(chemistry, mode)
                    view = self.single_view(chemistry, mode)
                    expected = app.display_table_df(view)
                    snapshot = app.history_snapshot_df(view)
                    pd.testing.assert_frame_equal(expected, snapshot)
                    self.assertFalse(any(app.is_admin_only_column(col) for col in snapshot.columns))
                    for base in app.chemistry_output_base_order(chemistry)[1:]:
                        column = app.display_name_for_output(base, chemistry)
                        self.assertNotEqual(snapshot.loc[1, column], "")
                        self.assertEqual(snapshot.loc[0, column] != "", base.startswith("Time_"))
                    if chemistry == "LFP Gen2A":
                        self.assertNotIn("Time_MaxT", snapshot.columns)
                        self.assertNotIn("Time_MaxPower", snapshot.columns)

                    with TemporaryDirectory() as folder:
                        path = Path(folder) / "history.csv"
                        with patch.object(app, "history_csv_path", return_value=path):
                            app.append_history(snapshot.iloc[:1], chemistry)
                            first = app.load_history(chemistry)
                            self.assertEqual(list(first.columns), list(snapshot.columns))
                            pd.testing.assert_frame_equal(snapshot.iloc[:1], first)
                            app.append_history(snapshot.iloc[1:], chemistry)
                            stored = pd.read_csv(path, dtype=str, keep_default_na=False)
                            pd.testing.assert_frame_equal(snapshot, stored)
                            pd.testing.assert_frame_equal(snapshot, app.load_history(chemistry))

    def test_mixed_saved_modes_do_not_recalculate(self):
        for chemistry in app.CHEMISTRY_OPTIONS:
            with self.subTest(chemistry=chemistry), TemporaryDirectory() as folder:
                path = Path(folder) / "history.csv"
                with patch.object(app, "history_csv_path", return_value=path):
                    snapshots = []
                    for mode in ["point", "band", "tolerance"]:
                        self.configure(chemistry, mode)
                        snapshot = app.history_snapshot_df(self.single_view(chemistry, mode))
                        snapshots.append(snapshot)
                        app.append_history(snapshot, chemistry)
                    app.st.session_state["admin_show_confidence_band_persist"] = True
                    app.st.session_state["admin_confidence_level_persist"] = 99
                    expected = pd.concat(snapshots, ignore_index=True)
                    pd.testing.assert_frame_equal(expected, app.load_history(chemistry))

    def test_legacy_blank_probability_bounds(self):
        self.configure("LFP Gen2B", "band")
        for mode in ["point", "tolerance"]:
            for saved_band in [False, True]:
                with self.subTest(mode=mode, saved_band=saved_band), TemporaryDirectory() as folder:
                    legacy = self.predictions("LFP Gen2B", mode)
                    legacy["p_tp_ci95_lo"] = ["", "0.40"]
                    legacy["p_tp_ci95_hi"] = ["", "0.60"]
                    if saved_band:
                        legacy["p_tp_ci95_band"] = ["", "0.40~0.60"]
                    if mode == "tolerance":
                        legacy["__p_tp_tol_lo"] = ""
                        legacy["__p_tp_tol_hi"] = ""
                    path = Path(folder) / "history.csv"
                    legacy.to_csv(path, index=False)
                    original_bytes = path.read_bytes()
                    with patch.object(app, "history_csv_path", return_value=path):
                        loaded = app.load_history("LFP Gen2B")
                    self.assertEqual(path.read_bytes(), original_bytes)
                    self.assertEqual(loaded.loc[0, "TP/NoTP"], "TP")
                    self.assertIn("0.40~0.60", loaded.loc[1, "TP/NoTP"])
                    self.assertEqual(loaded.loc[0, "MaxT_adj"], "")
                    self.assertNotEqual(loaded.loc[1, "MaxT_adj"], "")
                    self.assertNotEqual(loaded.loc[0, "Time_MaxT"], "")

    def test_legacy_ranges_survive_conversion_and_append(self):
        self.configure("LFP Gen2B", "band")
        point = self.predictions("LFP Gen2B", "point")
        legacy_range = self.predictions("LFP Gen2B", "tolerance").drop(columns=["__range_collapsed__"])
        legacy_range["A_Cell_D"] = "12.40<br>(+/-5.0%)"
        legacy = pd.concat([point, legacy_range], ignore_index=True)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "history.csv"
            legacy.to_csv(path, index=False)
            with patch.object(app, "history_csv_path", return_value=path):
                loaded = app.load_history("LFP Gen2B")
                self.assertEqual(loaded.loc[3, "MaxT_adj"], "30.00~40.00")
                self.assertEqual(loaded.loc[2, "MaxT_adj"], "")
                self.assertEqual(loaded.loc[2, "Time_MaxT"], "10.00~20.00")
                self.assertEqual(loaded.loc[3, "A_Cell_D"], "12.40<br>(+/-5.0%)")
                app.append_history(loaded.iloc[[3]], "LFP Gen2B")
                stored = pd.read_csv(path, dtype=str, keep_default_na=False)
                self.assertFalse(any(app.is_admin_only_column(col) for col in stored.columns))
                self.assertTrue(stored.columns.is_unique)
                self.assertEqual(stored.loc[4, "MaxT_adj"], "30.00~40.00")


if __name__ == "__main__":
    unittest.main()