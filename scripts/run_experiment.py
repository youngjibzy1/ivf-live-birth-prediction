#!/usr/bin/env python3
"""Run the IVF live-birth prediction experiment.

Usage::

    python scripts/run_experiment.py --seed 42 --cycles 2500
    python scripts/run_experiment.py --csv /path/to/cohort.csv
"""

from __future__ import annotations

import argparse
import json

from ivf_live_birth.config import ExperimentConfig
from ivf_live_birth.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycles", type=int, default=2500)
    parser.add_argument("--csv", default=None, help="path to real cohort CSV")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    cfg = ExperimentConfig(n_cycles=args.cycles, seed=args.seed)
    report = run_experiment(
        seed=args.seed,
        csv_path=args.csv,
        config=cfg,
        output_dir=args.out,
    )

    print("=" * 72)
    print("IVF live-birth prediction - experiment report")
    print("=" * 72)
    d, cv, tm = report["data"], report["cross_validation"], report["test_metrics"]
    print(f"Data source        : {d['source']}")
    print(f"Cycles             : {d['n_cycles']}  (live-birth rate {d['live_birth_rate']:.1%})")
    print(f"Cross-validation   : ROC-AUC {cv['cv_roc_auc_mean']:.3f} "
          f"(+/- {cv['cv_roc_auc_std']:.3f}, {cv['cv_folds']} folds)")
    print(f"Hold-out test set  : n={tm['n_test']}")
    print(f"  ROC-AUC          : {tm['roc_auc']:.3f}")
    print(f"  PR-AUC           : {tm['pr_auc']:.3f}")
    print(f"  Brier score      : {tm['brier_score']:.3f}")
    print(f"  Youden threshold : {tm['youden_threshold']:.3f} "
          f"(sens {tm['at_youden']['sensitivity']:.2f}, spec {tm['at_youden']['specificity']:.2f})")
    print("\nTop predictors (permutation importance):")
    for item in tm["permutation_importance"][:5]:
        print(f"  {item['feature']:<22} {item['mean']:+.4f}")

    print(f"\nFull report: {args.out}/report.json")


if __name__ == "__main__":
    main()