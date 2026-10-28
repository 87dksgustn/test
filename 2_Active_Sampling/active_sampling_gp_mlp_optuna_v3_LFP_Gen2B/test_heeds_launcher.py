import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from launcher_heeds import build_cumulative_dataset, create_parser, discover_input_files, execute, main


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = SimpleNamespace(
            BASE_CONTINUOUS_COLS=["Position"], DISCRETE_COLS=["Material"],
            TPNoTP_COL="TP", TMAX_COL="Tmax", OTHER_REGRESSION_COLS=["Extra"],
            TIME_FEATURE_COLS=[], PASS_LABEL=0, FAIL_LABEL=1,
            DISCRETE_LEVELS={"Material": ["Si", "PU"]},
        )
        self.initial = {"Eval#": 1, "Position": 1.0, "Material": "Si", "TP": 0, "Tmax": 100.0, "Extra": 2.0}
        self.write_source("Initial_Dataset.csv", [self.initial])

    def write_source(self, name, rows):
        pd.DataFrame(rows).to_csv(self.root / name, index=False, encoding="utf-8-sig")


class CumulativeDatasetTests(DatasetFixture):
    def test_numeric_order_and_exact_filename_matching(self):
        for name, position in [("Itr_10_Dataset.csv", 10), ("Itr_2_Dataset.csv", 2)]:
            self.write_source(name, [{**self.initial, "Position": position}])
        for name in ["Itr_11_Dataset - copy.csv", "Itr_12_Dataset.csv.tmp", "z_Final_Dataset.csv", "cumulative_dataset.csv"]:
            self.write_source(name, [self.initial])
        frame, report = build_cumulative_dataset(self.root, self.cfg)
        self.assertEqual(frame["Position"].tolist(), [1, 2, 10])
        self.assertEqual(frame["_source_file"].tolist(), ["Initial_Dataset.csv", "Itr_2_Dataset.csv", "Itr_10_Dataset.csv"])
        self.assertEqual(frame["_source_row"].tolist(), [2, 2, 2])
        self.assertEqual(report["source_rows"], 3)

    def test_rerun_rebuilds_without_appending(self):
        first, first_report = build_cumulative_dataset(self.root, self.cfg)
        first.to_csv(self.root / "cumulative_dataset.csv", index=False)
        second, second_report = build_cumulative_dataset(self.root, self.cfg)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_report, second_report)

    def test_initial_dataset_is_required(self):
        (self.root / "Initial_Dataset.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Initial dataset"):
            discover_input_files(self.root)

    def test_duplicate_iteration_numbers_are_rejected(self):
        self.write_source("Itr_1_Dataset.csv", [self.initial])
        self.write_source("Itr_01_Dataset.csv", [self.initial])
        with self.assertRaisesRegex(ValueError, "Duplicate iteration"):
            discover_input_files(self.root)

    def test_duplicate_rows_are_rejected_by_default(self):
        self.write_source("Itr_1_Dataset.csv", [{**self.initial, "Eval#": 9}])
        with self.assertRaisesRegex(ValueError, "Duplicate input conditions"):
            build_cumulative_dataset(self.root, self.cfg)

    def test_identical_exports_can_be_explicitly_deduplicated(self):
        self.write_source("Itr_1_Dataset.csv", [{**self.initial, "Eval#": 9, "Unnamed: 8": "Initial"}])
        frame, report = build_cumulative_dataset(self.root, self.cfg, "drop-identical")
        self.assertEqual(len(frame), 1)
        self.assertEqual(report["identical_rows_removed"], 1)
        self.assertNotIn("Unnamed: 8", frame.columns)

    def test_conflicting_results_are_never_silently_overwritten(self):
        self.write_source("Itr_1_Dataset.csv", [{**self.initial, "Tmax": 101.0}])
        with self.assertRaisesRegex(ValueError, "Conflicting results"):
            build_cumulative_dataset(self.root, self.cfg, "drop-identical")

    def test_invalid_sources_include_filename_in_error(self):
        for updates in [{"Position": None}, {"TP": 3}, {"Material": "Unknown"}, {"Tmax": "unfinished"}]:
            with self.subTest(updates=updates):
                self.write_source("Itr_1_Dataset.csv", [{**self.initial, **updates}])
                with self.assertRaisesRegex(ValueError, "Invalid dataset Itr_1_Dataset.csv"):
                    build_cumulative_dataset(self.root, self.cfg)

    def test_missing_target_column_is_rejected(self):
        self.write_source("Itr_1_Dataset.csv", [{key: value for key, value in self.initial.items() if key != "Extra"}])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            build_cumulative_dataset(self.root, self.cfg)

    def test_blank_regression_targets_are_allowed(self):
        self.write_source("Itr_1_Dataset.csv", [{**self.initial, "Position": 2, "TP": 1, "Tmax": None, "Extra": None}])
        frame, report = build_cumulative_dataset(self.root, self.cfg)
        self.assertTrue(pd.isna(frame.iloc[1]["Tmax"]))
        self.assertEqual(report["training_rows"], 2)


class LauncherTests(DatasetFixture):
    def setUp(self):
        super().setUp()
        self.output_dir = self.root / "results"
        self.cfg.INPUT_CSV = "legacy_input.csv"
        self.cfg.OUTPUT_DIR = Path("legacy_outputs")
        self.cfg.FINAL_TEST_CSV = "legacy_holdout.csv"
        self.original_config = dict(vars(self.cfg))

    def arguments(self, *extra):
        return create_parser().parse_args([
            "--data-dir", str(self.root), "--output-dir", str(self.output_dir), *extra,
        ])

    def fake_pipeline(self):
        self.assertEqual(self.cfg.OUTPUT_DIR, self.output_dir)
        self.assertIsNone(self.cfg.FINAL_TEST_CSV)
        training = pd.read_csv(self.cfg.INPUT_CSV)
        self.assertEqual(len(training), 1)
        self.assertIn("input_snapshots", str(self.cfg.INPUT_CSV))
        run_dir = self.cfg.OUTPUT_DIR / "Try_1"
        run_dir.mkdir()
        training.assign(sampling_rank=1).to_csv(run_dir / "next_sampling_candidates.csv", index=False)

    def test_prepare_only_does_not_call_training(self):
        with patch("launcher_heeds.run_pipeline") as pipeline:
            result = execute(self.arguments("--prepare-only"), self.cfg)
        pipeline.assert_not_called()
        self.assertEqual(result["status"], "prepared")
        self.assertTrue(Path(result["input_csv"]).is_file())
        self.assertFalse((self.output_dir / "next_sampling_candidates.csv").exists())
        self.assertFalse((self.output_dir / ".heeds.lock").exists())
        self.assertEqual(vars(self.cfg), self.original_config)

    def test_success_publishes_candidates_and_restores_config(self):
        entry = Mock(side_effect=self.fake_pipeline)
        with patch.dict("sys.modules", {"0_main_active_sampling": SimpleNamespace(main=entry)}):
            result = execute(self.arguments(), self.cfg)
        entry.assert_called_once_with()
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(Path(result["candidates_csv"]).is_file())
        self.assertEqual(json.loads((self.output_dir / "run_status.json").read_text()), result)
        self.assertEqual(vars(self.cfg), self.original_config)

    def test_pipeline_error_restores_config_and_returns_no_result(self):
        def failing_pipeline():
            self.assertNotEqual(self.cfg.INPUT_CSV, self.original_config["INPUT_CSV"])
            raise RuntimeError("Simulated training failure")

        with patch.dict("sys.modules", {"0_main_active_sampling": SimpleNamespace(main=failing_pipeline)}):
            with self.assertLogs("heeds", level="ERROR"):
                with self.assertRaisesRegex(RuntimeError, "Simulated training failure"):
                    execute(self.arguments(), self.cfg)
        self.assertEqual(vars(self.cfg), self.original_config)
        self.assertEqual(json.loads((self.output_dir / "run_status.json").read_text())["status"], "failed")
        self.assertFalse((self.output_dir / "next_sampling_candidates.csv").exists())
        self.assertFalse((self.output_dir / ".heeds.lock").exists())

    def test_failed_validation_removes_stale_candidate_csv(self):
        self.output_dir.mkdir()
        stale = self.output_dir / "next_sampling_candidates.csv"
        stale.write_text("stale result", encoding="utf-8")
        self.write_source("Itr_1_Dataset.csv", [self.initial])
        with self.assertLogs("heeds", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "Duplicate input"):
                execute(self.arguments(), self.cfg)
        self.assertFalse(stale.exists())

    def test_concurrent_run_is_rejected_without_changing_active_status(self):
        self.output_dir.mkdir()
        lock_path = self.output_dir / ".heeds.lock"
        lock_path.write_text("active process", encoding="utf-8")
        status_path = self.output_dir / "run_status.json"
        status_path.write_text("active status", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            execute(self.arguments(), self.cfg)
        self.assertEqual(status_path.read_text(), "active status")
        self.assertTrue(lock_path.exists())

    def test_missing_explicit_holdout_is_rejected(self):
        with self.assertLogs("heeds", level="ERROR"):
            with self.assertRaisesRegex(FileNotFoundError, "Holdout dataset"):
                execute(self.arguments("--holdout-csv", str(self.root / "missing.csv")), self.cfg)

    def test_training_source_cannot_be_the_holdout(self):
        with self.assertLogs("heeds", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "must not also be a training source"):
                execute(self.arguments("--holdout-csv", str(self.root / "Initial_Dataset.csv")), self.cfg)

    def test_cli_failure_returns_nonzero_exit_code(self):
        with self.assertLogs("heeds", level="ERROR"):
            exit_code = main(["--data-dir", str(self.root / "missing"), "--output-dir", str(self.output_dir)])
        self.assertEqual(exit_code, 1)


class LinuxBuildSpecTests(unittest.TestCase):
    def test_spec_bundles_shared_main_as_a_headless_console_app(self):
        root = Path(__file__).resolve().parent
        hooks = SimpleNamespace(
            copy_metadata=Mock(return_value=[]),
            collect_submodules=Mock(return_value=["optuna.samplers"]),
        )
        analysis = Mock(return_value=SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[]))
        executable = Mock()
        collect = Mock()
        modules = {
            "PyInstaller": SimpleNamespace(),
            "PyInstaller.utils": SimpleNamespace(),
            "PyInstaller.utils.hooks": hooks,
        }
        with patch.dict("sys.modules", modules), patch("sys.platform", "linux"):
            runpy.run_path(str(root / "HEEDSActiveSampler.spec"), init_globals={
                "SPECPATH": str(root), "Analysis": analysis,
                "PYZ": Mock(), "EXE": executable, "COLLECT": collect,
            })
        self.assertEqual(analysis.call_args.args[0], [str(root / "launcher_heeds.py")])
        self.assertIn("0_main_active_sampling", analysis.call_args.kwargs["hiddenimports"])
        self.assertIn("streamlit", analysis.call_args.kwargs["excludes"])
        self.assertEqual(analysis.call_args.kwargs["hooksconfig"]["matplotlib"]["backends"], ["Agg"])
        self.assertTrue(executable.call_args.kwargs["console"])
        self.assertTrue(executable.call_args.kwargs["exclude_binaries"])
        self.assertEqual(collect.call_args.kwargs["name"], "heeds_active_sampler")


if __name__ == "__main__":
    unittest.main()