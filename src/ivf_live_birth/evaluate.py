"""Evaluation: clinical metrics, threshold selection and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Operating point maximising sensitivity + specificity (Youden's J)."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thresholds[best])


def metrics_at_threshold(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": round(threshold, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "accuracy": round((tn + tp) / max(len(y_true), 1), 4),
        "n_predicted_positive": int(preds.sum()),
    }


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    output_dir: Path,
) -> dict:
    """Full evaluation: metrics, operating point, permutation importance, plots."""

    y = y_test.to_numpy(dtype=int)
    proba = pipeline.predict_proba(X_test)[:, 1]
    threshold = youden_threshold(y, proba)

    report = {
        "n_test": int(len(y)),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "brier_score": round(float(brier_score_loss(y, proba)), 4),
        "youden_threshold": round(threshold, 4),
        "at_youden": metrics_at_threshold(y, proba, threshold),
        # Clinical convention: fix specificity ~= 60% and report sensitivity.
        "at_specificity_60": {
            k: v for k, v in metrics_at_threshold_spec(y, proba, 0.60).items()
        },
    }

    # ROC / PR / calibration plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fpr, tpr, _ = roc_curve(y, proba)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC {report['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="ROC curve", xlabel="1 - specificity", ylabel="sensitivity")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y, proba)
    axes[1].plot(rec, prec, lw=2, label=f"PR-AUC {report['pr_auc']:.3f}")
    axes[1].set(title="Precision-Recall", xlabel="recall", ylabel="precision")
    axes[1].legend()

    prob_positive, prob_pred = calibration_curve(y, proba, n_bins=8)
    axes[2].plot(prob_pred, prob_positive, "o-", label="model")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    axes[2].set(title="Calibration (Brier %.3f)" % report["brier_score"],
                xlabel="mean predicted", ylabel="fraction positive")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "model_metrics.png", dpi=120)
    plt.close(fig)

    # Permutation importance (model-agnostic)
    perm = permutation_importance(
        pipeline, X_test, y, n_repeats=10, random_state=7, scoring="roc_auc"
    )
    imps = sorted(
        zip(feature_cols, perm.importances_mean, perm.importances_std),
        key=lambda t: -t[1],
    )
    report["permutation_importance"] = [
        {"feature": f, "mean": round(m, 4), "std": round(s, 4)}
        for f, m, s in imps
    ]

    # Save test predictions for the audit trail
    pd.DataFrame({
        "predicted_probability": proba,
        "actual_live_birth": y,
    }).to_csv(output_dir / "test_predictions.csv", index=False)

    return report


def metrics_at_threshold_spec(
    y_true: np.ndarray, proba: np.ndarray, target_spec: float
) -> dict:
    """Highest-sensitivity operating point achieving >= target specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    viable = [
        (tid, thr) for tid, thr in enumerate(thresholds) if 1 - fpr[tid] >= target_spec
    ]
    best_tid, best_thr = max(viable, key=lambda p: tpr[p[0]])
    report = metrics_at_threshold(y_true, proba, best_thr)
    report["target_specificity"] = target_spec
    return report


def write_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2))