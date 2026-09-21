"""End-to-end experiment test."""

from __future__ import annotations

import json
from pathlib import Path

from ivf_live_birth.config import ExperimentConfig
from ivf_live_birth.experiment import run_experiment


def test_experiment_runs_end_to_end(tmp_path: Path):
    cfg = ExperimentConfig(n_cycles=400, output_dir=str(tmp_path))
    report = run_experiment(seed=1, config=cfg)

    assert report["data"]["n_cycles"] == 400
    assert report["cross_validation"]["cv_roc_auc_mean"] > 0.5
    assert report["test_metrics"]["roc_auc"] > 0.5

    for f in ("ivf_model.joblib", "report.json", "model_metrics.png",
              "test_predictions.csv"):
        assert (tmp_path / f).exists(), f"missing {f}"

    # report.json parses and is complete
    parsed = json.loads((tmp_path / "report.json").read_text())
    assert "permutation_importance" in parsed["test_metrics"]
    assert parsed["test_metrics"]["at_youden"]["sensitivity"] >= 0
    assert parsed["test_metrics"]["at_youden"]["sensitivity"] <= 1


def test_experiment_evaluates_real_csv(tmp_path: Path):
    import pandas as pd

    df = pd.read_csv("data/sample_cohort.csv") if Path("data/sample_cohort.csv").exists() else None
    if df is None:
        return  # data asset optional in source checkout
    cfg = ExperimentConfig(n_cycles=len(df), output_dir=str(tmp_path))
    report = run_experiment(csv_path="data/sample_cohort.csv", config=cfg)
    assert report["data"]["source"] == "data/sample_cohort.csv"