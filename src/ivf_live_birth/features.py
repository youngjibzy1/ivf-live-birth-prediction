"""Feature engineering and dataset splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def prepare_data(
    df: pd.DataFrame,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """One-hot encode categoricals, stratify and split train/test."""

    cfg = config or ExperimentConfig()

    data = df.copy()

    # One-hot the cause of infertility. We drop the *first category from the
    # config order* (not pandas' lexicographic order) so the reference class
    # is deterministic.
    data = pd.get_dummies(data, columns=["cause"], prefix="cause", dtype=int)
    data = data.drop(
        columns=[f"cause_{cfg.cause_categories[0]}"], errors="ignore"
    )

    feature_cols = list(cfg.feature_cols)
    onehot_cols = [
        c for c in data.columns if c.startswith("cause_")
    ]
    feature_cols += onehot_cols

    X = data[feature_cols]
    y = data[cfg.outcome_col].astype(int)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.seed,
    )
    return X_train, y_train, X_test, y_test


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick descriptive stats for the report (EDA table)."""
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=["patient_id", "true_p_live_birth"], errors="ignore"
    )
    return numeric.describe().T.round(3)