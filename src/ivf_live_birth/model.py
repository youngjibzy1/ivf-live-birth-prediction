"""Model training: gradient boosting + logistic-regression baseline.

The primary model is scikit-learn's HistGradientBoostingClassifier (fast,
handles mixed tabular features, no missing-data headaches); the logistic
regression is the interpretable clinical baseline. Both are wrapped in small
pipelines so real CSV data plugs straight in.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


def build_hgb_pipeline(config: ExperimentConfig | None = None) -> Pipeline:
    cfg = config or ExperimentConfig()
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=cfg.hgb_max_iter,
            learning_rate=cfg.hgb_learning_rate,
            max_depth=cfg.hgb_max_depth,
            random_state=cfg.seed,
        )),
    ])


def build_lr_pipeline(config: ExperimentConfig | None = None) -> Pipeline:
    cfg = config or ExperimentConfig()
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=cfg.seed)),
    ])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kind: str = "hgb",
    config: ExperimentConfig | None = None,
) -> Pipeline:
    pipeline = build_hgb_pipeline(config) if kind == "hgb" else build_lr_pipeline(config)
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ExperimentConfig | None = None,
) -> dict:
    """5-fold stratified CV of the gradient boosting model."""

    cfg = config or ExperimentConfig()
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    aucs: list[float] = []

    X = X_train.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=int)

    for train_idx, val_idx in cv.split(X, y):
        pipe = build_hgb_pipeline(cfg)
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], proba))

    return {
        "cv_roc_auc_mean": float(np.mean(aucs)),
        "cv_roc_auc_std": float(np.std(aucs)),
        "cv_folds": cfg.cv_folds,
    }


def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)