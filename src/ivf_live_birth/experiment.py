"""End-to-end experiment: data -> model -> evaluation report.

Runs the whole pipeline with a single call, writing everything to ``output/``:

* ``ivf_model.joblib``   - serialised gradient-boosting model
* ``report.json``        - metrics, operating point, permutation importance
* ``model_metrics.png``  - ROC / PR / calibration curves
* ``test_predictions.csv`` - per-patient predictions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data_generator import generate_cohort
from .evaluate import evaluate, write_report
from .features import feature_summary, prepare_data
from .model import cross_validate, save_model, train_model


def run_experiment(
    seed: int | None = None,
    csv_path: str | None = None,
    config: ExperimentConfig | None = None,
    output_dir: str | None = None,
) -> dict:
    cfg = config or ExperimentConfig()
    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data (synthetic by default; real CSV drops in below).
    if csv_path:
        cohort = pd.read_csv(csv_path)
        if "cause" not in cohort.columns:
            cohort["cause"] = "unexplained"
    else:
        cohort = generate_cohort(config=cfg, seed=seed)

    # 2. Split
    X_train, y_train, X_test, y_test = prepare_data(cohort, cfg)
    feature_cols = list(X_train.columns)

    # 3. Cross-validation (honest estimate)
    cv = cross_validate(X_train, y_train, cfg)

    # 4. Train final model on all training data
    model = train_model(X_train, y_train, kind="hgb", config=cfg)
    save_model(model, str(out / "ivf_model.joblib"))

    # 5. Evaluate on the held-out test set
    metrics = evaluate(model, X_test, y_test, feature_cols, out)

    # 6. Assemble report
    report = {
        "data": {
            "source": "synthetic-calibrated" if not csv_path else csv_path,
            "n_cycles": int(len(cohort)),
            "live_birth_rate": round(float(cohort["live_birth"].mean()), 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "cross_validation": cv,
        "test_metrics": metrics,
        "feature_summary": feature_summary(cohort).to_dict(),
    }
    write_report(report, out / "report.json")
    print(f"Experiment complete -> {out}/")
    return report