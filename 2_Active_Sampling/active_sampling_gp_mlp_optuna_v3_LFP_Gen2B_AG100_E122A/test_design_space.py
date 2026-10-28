import csv
import importlib
import importlib.util
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import make_final_test_doe
import make_holdout_distribution_plots
import make_holdout_doe_from_reference_dist
import predict_surrogate
from batch_selector import bin_label_for_value
from candidate_generator import generate_candidate_pool
from data_loader import load_candidate_data, load_labeled_data
from discrete_space import generate_valid_discrete_combinations
from preprocessing import build_preprocessor, make_xy
from surrogate_bundle import build_config_snapshot, make_feature_frame


class DesignSpaceTests(unittest.TestCase):
    def setUp(self):
        self.combos = generate_valid_discrete_combinations(
            config.DISCRETE_LEVELS, config.DISCRETE_COLS, config.S_PREFIX
        )
        self.reference = generate_candidate_pool(
            self.combos, config.BASE_CONTINUOUS_COLS, config.CONTINUOUS_BOUNDS,
            config.DISCRETE_COLS, 50, {}, config.RANDOM_SEED,
        )
        self.reference[config.TPNoTP_COL] = np.arange(len(self.reference)) % 2
        self.reference[config.TMAX_COL] = 80 + self.reference["G_PreForce"] * 0.01

    def assert_design_valid(self, frame):
        for column, (lower, upper) in config.CONTINUOUS_BOUNDS.items():
            self.assertTrue(frame[column].between(lower, upper).all(), column)
        for column, spec in config.EXCLUDED_REFERENCE_RANGES.items():
            self.assertFalse(frame[column].between(
                spec["center"] - spec["half_width"], spec["center"] + spec["half_width"]
            ).any(), column)
        for first_col, second_col, interaction_col in config.INTERACTION_TERMS:
            np.testing.assert_allclose(frame[interaction_col], frame[first_col] * frame[second_col])
        for column, levels in config.DISCRETE_LEVELS.items():
            self.assertTrue(set(frame[column]).issubset(levels), column)

    def test_labeled_csv_whitespace_and_center_conversion(self):
        raw = self.reference.iloc[:3].copy()
        for column in config.BASE_CONTINUOUS_COLS:
            if column != "G_PreForce":
                raw[column] -= config.EXCLUDED_REFERENCE_RANGES[column]["center"]
        raw = raw.map(lambda value: f"  {value}  " if isinstance(value, str) else value)
        raw["B_Barrier_Type"] = " A 1 "
        raw.columns = [f"  {column}  " for column in raw.columns]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            raw.to_csv(path, index=False)
            loaded = load_labeled_data(path, config)
            for column in config.BASE_CONTINUOUS_COLS:
                np.testing.assert_allclose(loaded[column], self.reference.iloc[:3][column])
            self.assertTrue(loaded["B_Barrier_Type"].eq("A1").all())
            self.assertFalse(any(any(char.isspace() for char in column) for column in loaded.columns))
            raw.drop(columns=["  G_PreForce  "]).to_csv(path, index=False)
            self.assertNotIn("G_PreForce", load_labeled_data(path, config))

    def test_config_and_derived_bounds(self):
        defaults = build_config_snapshot(config)["continuous_defaults"]
        self.assertEqual(defaults, {
            column: spec["center"]
            for column, spec in config.EXCLUDED_REFERENCE_RANGES.items()
            if column in config.BASE_CONTINUOUS_COLS
        })
        self.assertEqual(defaults["G_PreForce"], 100.0)
        self.assertEqual(config.CURRENT_LEVEL_TARGET_TOTAL, config.INITIAL_TOTAL + config.ADDITIONAL_TOTAL)
        for first_col, second_col, interaction_col in config.INTERACTION_TERMS:
            products = [
                first_bound * second_bound
                for first_bound in config.CONTINUOUS_BOUNDS[first_col]
                for second_bound in config.CONTINUOUS_BOUNDS[second_col]
            ]
            self.assertEqual(config.CONTINUOUS_BOUNDS[interaction_col], (min(products), max(products)))

    def test_training_inputs_and_candidates(self):
        snapshot = build_config_snapshot(config)
        features = make_feature_frame(self.reference, snapshot)
        labeled = features.assign(**{
            config.TPNoTP_COL: self.reference[config.TPNoTP_COL],
            config.TMAX_COL: self.reference[config.TMAX_COL],
        })
        x_raw, _, _ = make_xy(labeled, config.CONTINUOUS_COLS, config.DISCRETE_COLS, config.TPNoTP_COL, config.TMAX_COL)
        preprocessor = build_preprocessor(config.CONTINUOUS_COLS, config.DISCRETE_COLS)
        x_train = preprocessor.fit_transform(x_raw)
        self.assertIn("continuous__G_PreForce", preprocessor.get_feature_names_out())
        self.assertTrue(np.isfinite(x_train).all())
        candidates = generate_candidate_pool(
            self.combos, config.BASE_CONTINUOUS_COLS, config.CONTINUOUS_BOUNDS,
            config.DISCRETE_COLS, 128, config.EXCLUDED_REFERENCE_RANGES, config.RANDOM_SEED,
        )
        candidate_features = make_feature_frame(candidates, snapshot)
        self.assert_design_valid(candidate_features)
        self.assertEqual(preprocessor.transform(candidate_features).shape[1], x_train.shape[1])

    def test_cell_d_bins_cover_design_range(self):
        lower, upper = config.CONTINUOUS_BOUNDS["A_Cell_D"]
        for prefix in ("TP_RATIO", "NOTP_HIGHTMAX", "HYBRID"):
            bins = getattr(config, f"{prefix}_CELLD_BINS")
            labels = getattr(config, f"{prefix}_CELLD_BIN_LABELS")
            self.assertEqual(len(bins), len(labels) + 1)
            for value in np.linspace(lower, upper, 181):
                self.assertIsNotNone(bin_label_for_value(value, bins, labels), (prefix, value))

    def test_direct_prediction_inputs(self):
        arguments = []
        for column in config.BASE_CONTINUOUS_COLS:
            if column != "G_PreForce":
                arguments.extend([f"--{column}", str(config.CONTINUOUS_BOUNDS[column][0])])
        for column in config.DISCRETE_COLS:
            arguments.extend([f"--{column}", config.DISCRETE_LEVELS[column][0]])
        parser = predict_surrogate.build_parser()
        default_row = predict_surrogate.build_single_input_df(parser.parse_args(arguments))
        self.assertEqual(default_row.iloc[0]["G_PreForce"], 100.0)
        explicit_row = predict_surrogate.build_single_input_df(parser.parse_args(arguments + ["--G_PreForce", "900"]))
        self.assertEqual(explicit_row.iloc[0]["G_PreForce"], 900.0)
        default_arguments = []
        for column in config.DISCRETE_COLS:
            default_arguments.extend([f"--{column}", config.DISCRETE_LEVELS[column][0]])
        all_defaults = predict_surrogate.build_single_input_df(parser.parse_args(default_arguments))
        for column in config.BASE_CONTINUOUS_COLS:
            self.assertEqual(all_defaults.iloc[0][column], config.EXCLUDED_REFERENCE_RANGES[column]["center"])

    def test_final_doe_from_base_only_reference(self):
        final, boundary, coverage, weights = make_final_test_doe.build_test_dataset(self.reference, self.combos)
        self.assertEqual(len(final), make_final_test_doe.TOTAL_TEST_CASES)
        self.assert_design_valid(final)
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "OUTPUT_DIR", Path(directory)):
            dataset_path, summary_path, report_path = make_final_test_doe.save_outputs(final, boundary, coverage, weights)
            self.assert_design_valid(pd.read_csv(dataset_path))
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())

    def test_reference_doe_preforce_alias_and_exclusions(self):
        generator = make_holdout_doe_from_reference_dist
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.csv"
            self.reference.rename(columns={"G_PreForce": "PreForce"}).to_csv(reference_path, index=False)
            with patch.object(generator, "REFERENCE_CSV", reference_path), patch.object(config, "OUTPUT_DIR", Path(directory)), patch("sys.argv", ["make_holdout_doe_from_reference_dist.py", "--n-output", "8"]):
                generator.main()
            generated = pd.read_csv(Path(directory) / "final_test_doe" / "holdout_doe_8_from_initial_distribution.csv")
            self.assertEqual(len(generated), 8)
            self.assert_design_valid(generated)

    def test_candidate_csv_layout_and_offsets(self):
        sampling = importlib.import_module("0_main_active_sampling")
        selected = self.reference.iloc[:3].copy()
        for column in config.BASE_CONTINUOUS_COLS:
            lower, upper = config.CONTINUOUS_BOUNDS[column]
            center = config.EXCLUDED_REFERENCE_RANGES.get(column, {}).get("center", 100.0)
            selected[column] = [center, lower, upper]
        selected = sampling.add_interaction_terms(selected, config.INTERACTION_TERMS)
        selected.insert(0, "sampling_rank", [1, 2, 3])
        selected["selected_bucket"] = "boundary"
        selected["p_tp"] = [0.45, 0.50, 0.55]
        original = selected.copy(deep=True)
        exported = sampling.make_candidate_export_frame(selected, config)
        input_cols = ["A_Cell_D", "B_Barrier_Type", "C_Barrier_Thx", "D_Barrier_Outer_Type", "E_Barrier_Outer_Thx", "F_ThermalResin_Thx", "G_PreForce"]
        remaining_cols = [column for column in selected.columns if column not in input_cols]
        self.assertEqual(exported.columns.tolist(), input_cols + [""] + remaining_cols)
        self.assertTrue(exported[""].eq("").all())
        for column in config.EXCLUDED_REFERENCE_RANGES:
            if column in config.BASE_CONTINUOUS_COLS and column != "G_PreForce":
                np.testing.assert_allclose(exported[column], selected[column] - config.EXCLUDED_REFERENCE_RANGES[column]["center"])
                self.assertEqual(exported.iloc[0][column], 0.0)
        pd.testing.assert_series_equal(exported["G_PreForce"], selected["G_PreForce"])
        pd.testing.assert_frame_equal(exported[remaining_cols], selected[remaining_cols])
        pd.testing.assert_frame_equal(selected, original)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "next_sampling_candidates.csv"
            exported.to_csv(output_path, index=False, encoding="utf-8-sig")
            with output_path.open(encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.reader(csv_file))
            self.assertEqual(rows[0], input_cols + [""] + remaining_cols)
            self.assertTrue(all(row[7] == "" for row in rows))
            self.assertEqual([float(row[6]) for row in rows[1:]], [100.0, 0.0, 1000.0])
            restored = load_candidate_data(output_path, config)
            pd.testing.assert_frame_equal(restored, original[input_cols + remaining_cols])
            legacy_path = Path(directory) / "legacy_candidates.csv"
            original.to_csv(legacy_path, index=False, encoding="utf-8-sig")
            pd.testing.assert_frame_equal(load_candidate_data(legacy_path, config), original)

    def test_sampling_dashboard_has_all_pairs(self):
        sampling = importlib.import_module("0_main_active_sampling")
        selected = self.reference.iloc[:5].copy()
        selected["selected_bucket"] = list(config.BUCKET_RATIO)
        selected["sampling_rank"] = range(1, len(selected) + 1)
        try:
            with patch.object(sampling.plt, "close"), patch("matplotlib.figure.Figure.savefig"):
                sampling.save_continuous_pairs_dashboard(self.reference, selected, config.BASE_CONTINUOUS_COLS, "unused.png")
                plotted = {(axis.get_xlabel(), axis.get_ylabel()) for axis in plt.gcf().axes if axis.get_visible()}
                self.assertEqual(plotted, set(itertools.combinations(config.BASE_CONTINUOUS_COLS, 2)))
        finally:
            plt.close("all")

    def test_holdout_scatter_outputs_cover_all_pairs(self):
        plots = make_holdout_distribution_plots
        with tempfile.TemporaryDirectory() as directory, patch.object(plots, "OUTPUT_DIR", Path(directory)), patch.object(plots.pd, "read_csv", return_value=self.reference.copy()), patch.object(plots, "save_combo_distribution"), patch.object(plots, "save_continuous_distribution") as distribution, patch.object(plots, "save_scatter_cellD_vs_barrier") as scatter, patch("sys.argv", ["make_holdout_distribution_plots.py"]):
            plots.main()
        self.assertEqual(distribution.call_args.args[3], config.BASE_CONTINUOUS_COLS)
        pairs = {tuple(call.args[3]) for call in scatter.call_args_list}
        self.assertEqual(pairs, set(itertools.combinations(config.BASE_CONTINUOUS_COLS, 2)))

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna is optional")
    def test_optimization_stays_inside_design_bounds(self):
        optimizer = importlib.import_module("z_surrogate_optimize_gp")
        bins = config.TP_RATIO_CELLD_BINS
        centers = [(lower + upper) / 2 for lower, upper in zip(bins[:-1], bins[1:])]
        reference = pd.DataFrame({"A_Cell_D": centers, config.TPNoTP_COL: config.TP_LABEL})
        rule = optimizer.build_celld_trial_quota(reference, config, len(centers))

        def choose_upper(name, low, high):
            design_lower, design_upper = config.CONTINUOUS_BOUNDS[name]
            self.assertGreaterEqual(low, design_lower)
            self.assertLessEqual(high, design_upper)
            return high

        def evaluate_stub(preprocessor, nn_model, models, cfg, combo_row, values):
            return {**values, "p_tp": 0.4, "tmax_pred_given_notp": 60.0, "constraint_violation": 0.0}

        with patch.object(optimizer.optuna.trial.Trial, "suggest_float", side_effect=choose_upper), patch.object(optimizer, "evaluate_point", side_effect=evaluate_stub):
            result = optimizer.optimize_single_combo(None, None, None, config, self.combos.iloc[0], len(centers), rule)
        self.assertEqual(len(result), len(centers))
        self.assertEqual(result["A_Cell_D"].max(), config.CONTINUOUS_BOUNDS["A_Cell_D"][1])
        self.assertIn("G_PreForce", result)


if __name__ == "__main__":
    unittest.main()