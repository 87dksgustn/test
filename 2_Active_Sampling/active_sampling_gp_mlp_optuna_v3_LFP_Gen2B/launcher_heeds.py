import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import re
import socket
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_labeled_data, validate_required_columns, validate_passfail_labels


ITERATION_PATTERN = re.compile(r"Itr_([0-9]+)_Dataset\.csv")
SOURCE_COLUMNS = ["_source_file", "_source_row"]
LOGGER = logging.getLogger("heeds")


def discover_input_files(data_dir):
    data_dir = Path(data_dir).resolve()
    initial = data_dir / "Initial_Dataset.csv"
    if not initial.is_file():
        raise FileNotFoundError(f"Initial dataset not found: {initial}")

    iterations = {}
    for path in data_dir.iterdir():
        match = ITERATION_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        iteration = int(match.group(1))
        if iteration in iterations:
            raise ValueError(f"Duplicate iteration {iteration}: {iterations[iteration].name}, {path.name}")
        iterations[iteration] = path
    return [initial] + [iterations[iteration] for iteration in sorted(iterations)]


def build_cumulative_dataset(data_dir, cfg, duplicate_policy="error"):
    if duplicate_policy not in {"error", "drop-identical"}:
        raise ValueError(f"Unknown duplicate policy: {duplicate_policy}")
    input_cols = list(cfg.BASE_CONTINUOUS_COLS) + list(cfg.DISCRETE_COLS)
    target_cols = list(dict.fromkeys(
        [cfg.TPNoTP_COL, cfg.TMAX_COL] + list(cfg.OTHER_REGRESSION_COLS) + list(cfg.TIME_FEATURE_COLS)
    ))
    required_cols = list(dict.fromkeys(input_cols + target_cols))
    frames = []
    sources = []
    for path in discover_input_files(data_dir):
        original_bytes = path.read_bytes()
        frame = load_labeled_data(path)
        if original_bytes != path.read_bytes():
            raise ValueError(f"Input changed while reading: {path}. Finish exporting before starting the launcher.")
        if frame.empty:
            raise ValueError(f"Input contains no data rows: {path}")
        if any(column in frame.columns for column in SOURCE_COLUMNS):
            raise ValueError(f"Reserved source columns found in {path.name}: {SOURCE_COLUMNS}")
        try:
            validate_required_columns(
                frame, cfg.BASE_CONTINUOUS_COLS, cfg.DISCRETE_COLS, cfg.TPNoTP_COL,
                cfg.TMAX_COL, cfg.OTHER_REGRESSION_COLS, cfg.TIME_FEATURE_COLS,
            )
            for column in list(cfg.BASE_CONTINUOUS_COLS) + target_cols:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
                values = frame[column].to_numpy(dtype=float)
                if np.isinf(values).any():
                    raise ValueError(f"Infinite values in {column}")
            if frame[input_cols + [cfg.TPNoTP_COL]].isna().any().any():
                raise ValueError("Input features and classification labels must not be blank")
            validate_passfail_labels(frame, cfg.TPNoTP_COL, cfg.PASS_LABEL, cfg.FAIL_LABEL)
            for column in cfg.DISCRETE_COLS:
                invalid = ~frame[column].isin(cfg.DISCRETE_LEVELS[column])
                if invalid.any():
                    raise ValueError(f"Invalid levels in {column}: {frame.loc[invalid, column].unique().tolist()}")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid dataset {path.name}: {error}") from error

        frame = frame.loc[:, [column for column in frame.columns if not str(column).startswith("Unnamed:")]].copy()
        frame["_source_file"] = path.name
        frame["_source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
        sources.append({
            "file": path.name,
            "path": str(path),
            "rows": int(len(frame)),
            "sha256": hashlib.sha256(original_bytes).hexdigest(),
        })

    combined = pd.concat(frames, ignore_index=True, sort=False)
    observations = combined.drop_duplicates(subset=required_cols)
    conflicts = observations.duplicated(subset=input_cols, keep=False)
    if conflicts.any():
        examples = observations.loc[conflicts, SOURCE_COLUMNS].head(10).to_dict("records")
        raise ValueError(f"Conflicting results for the same input conditions: {examples}")

    duplicates = combined.duplicated(subset=input_cols, keep="first")
    duplicate_count = int(duplicates.sum())
    if duplicate_count and duplicate_policy == "error":
        examples = combined.loc[duplicates, SOURCE_COLUMNS].head(10).to_dict("records")
        raise ValueError(
            f"Duplicate input conditions ({duplicate_count} rows): {examples}. "
            "Use --duplicates drop-identical only for repeated exports of the same observations."
        )
    combined = combined.loc[~duplicates].reset_index(drop=True)
    report = {
        "sources": sources,
        "source_rows": sum(source["rows"] for source in sources),
        "training_rows": int(len(combined)),
        "identical_rows_removed": duplicate_count,
        "duplicate_policy": duplicate_policy,
    }
    return combined, report


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8"))


@contextmanager
def output_lock(output_dir):
    lock_path = output_dir / ".heeds.lock"
    try:
        stream = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Output directory is locked: {lock_path}. "
            "Remove a stale lock only after verifying the previous process has stopped."
        ) from error
    try:
        with stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


@contextmanager
def runtime_config(cfg, overrides):
    original = {key: getattr(cfg, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(cfg, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


def run_pipeline(cfg, input_csv, output_dir, holdout_csv):
    import matplotlib

    matplotlib.use("Agg")
    overrides = {
        "INPUT_CSV": str(input_csv),
        "OUTPUT_DIR": output_dir,
        "FINAL_TEST_CSV": str(holdout_csv) if holdout_csv else None,
    }
    previous_runs = {path for path in output_dir.iterdir() if path.is_dir() and re.fullmatch(r"Try_[0-9]+", path.name)}
    with runtime_config(cfg, overrides):
        pipeline = importlib.import_module("0_main_active_sampling")
        pipeline.main()
    new_runs = {
        path for path in output_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"Try_[0-9]+", path.name)
    } - previous_runs
    if len(new_runs) != 1:
        raise RuntimeError(f"Expected one new Try directory, found {len(new_runs)}")
    run_dir = new_runs.pop()
    candidate_path = run_dir / "next_sampling_candidates.csv"
    candidates = load_labeled_data(candidate_path)
    required = ["sampling_rank"] + list(cfg.BASE_CONTINUOUS_COLS) + list(cfg.DISCRETE_COLS)
    if candidates.empty or any(column not in candidates.columns for column in required):
        raise ValueError(f"Pipeline produced an empty or invalid candidate CSV: {candidate_path}")
    return candidate_path


def execute(args, cfg):
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if data_dir == output_dir:
        raise ValueError("Use separate input and output directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(output_dir):
        handler = logging.FileHandler(output_dir / "heeds.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(handler)
        status_path = output_dir / "run_status.json"
        published_candidates = output_dir / "next_sampling_candidates.csv"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        status = {"run_id": run_id, "status": "running", "data_dir": str(data_dir), "output_dir": str(output_dir)}
        try:
            published_candidates.unlink(missing_ok=True)
            write_json(status_path, status)
            LOGGER.info("Starting HEEDS run %s | data=%s | output=%s", run_id, data_dir, output_dir)
            frame, report = build_cumulative_dataset(data_dir, cfg, args.duplicates)
            holdout_csv = Path(args.holdout_csv).resolve() if args.holdout_csv else None
            if holdout_csv is not None:
                if not holdout_csv.is_file():
                    raise FileNotFoundError(f"Holdout dataset not found: {holdout_csv}")
                if holdout_csv in {Path(source["path"]).resolve() for source in report["sources"]}:
                    raise ValueError("The holdout CSV must not also be a training source")
            snapshot_dir = output_dir / "input_snapshots" / run_id
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            input_csv = snapshot_dir / "cumulative_dataset.csv"
            input_bytes = frame.to_csv(index=False).encode("utf-8-sig")
            atomic_write(input_csv, input_bytes)
            write_json(snapshot_dir / "dataset_report.json", report)
            atomic_write(output_dir / "cumulative_dataset.csv", input_bytes)
            write_json(output_dir / "dataset_report.json", report)
            status.update({"input_csv": str(input_csv), "training_rows": report["training_rows"], "source_count": len(report["sources"])})
            write_json(status_path, status)
            LOGGER.info("Prepared %s training rows from %s source files", len(frame), len(report["sources"]))
            if args.prepare_only:
                status["status"] = "prepared"
            else:
                candidate_path = run_pipeline(cfg, input_csv, output_dir, holdout_csv)
                atomic_write(published_candidates, candidate_path.read_bytes())
                status.update({"status": "succeeded", "run_dir": str(candidate_path.parent), "candidates_csv": str(published_candidates)})
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, status)
            LOGGER.info("HEEDS run %s: %s", run_id, status["status"])
            return status
        except (Exception, KeyboardInterrupt) as error:
            published_candidates.unlink(missing_ok=True)
            status.update({"status": "failed", "error": str(error), "finished_at": datetime.now(timezone.utc).isoformat()})
            try:
                write_json(status_path, status)
            except OSError:
                LOGGER.exception("Could not save failure status")
            LOGGER.exception("HEEDS run %s failed", run_id)
            raise
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()


def check_runtime():
    import matplotlib

    matplotlib.use("Agg")
    versions = {"python": platform.python_version(), "platform": platform.platform(), "libc": platform.libc_ver()}
    for name in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "optuna", "torch"]:
        module = importlib.import_module(name)
        versions[name] = module.__version__
    importlib.import_module("0_main_active_sampling")
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessClassifier
    import torch
    from models_mlp import MultiHeadMLP

    features = qmc.LatinHypercube(d=2, seed=42).random(6)
    classifier = GaussianProcessClassifier(optimizer=None).fit(features, [0, 0, 0, 1, 1, 1])
    if not np.isfinite(classifier.predict_proba(features)).all():
        raise RuntimeError("GP runtime check failed")
    model = MultiHeadMLP(2, [4], dropout=0.0)
    logits, tmax, extra = model(torch.from_numpy(features.astype(np.float32)))
    if not np.isfinite(logits.detach().numpy()).all():
        raise RuntimeError("MLP runtime check failed")
    versions["runtime_check"] = "passed"
    return versions


def create_parser():
    parser = argparse.ArgumentParser(description="Merge HEEDS batch exports and run the existing active-sampling pipeline.")
    parser.add_argument("--data-dir", type=Path, help="Folder containing Initial_Dataset.csv and incremental Itr_<number>_Dataset.csv exports")
    parser.add_argument("--output-dir", type=Path, default=Path("heeds_outputs"), help="Dedicated HEEDS output folder (default: ./heeds_outputs)")
    parser.add_argument("--holdout-csv", type=Path, help="Optional independent holdout CSV; never merged into training data")
    parser.add_argument("--duplicates", choices=["error", "drop-identical"], default="error", help="Repeated input conditions are rejected unless identical exports are explicitly allowed")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and save cumulative data without fitting models or publishing candidates")
    parser.add_argument("--check-runtime", action="store_true", help="Check bundled dependencies and small GP/MLP operations, then exit")
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if not args.check_runtime and args.data_dir is None:
        parser.error("--data-dir is required unless --check-runtime is used")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        if args.check_runtime:
            result = check_runtime()
        else:
            import config

            result = execute(args, config)
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    except Exception as error:
        LOGGER.error("HEEDS execution failed: %s", error)
        return 1


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    sys.exit(main())