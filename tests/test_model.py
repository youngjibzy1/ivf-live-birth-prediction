"""Model + evaluation tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ivf_live_birth.config import ExperimentConfig
from ivf_live_birth.data_generator import generate_cohort
from ivf_live_birth.features import prepare_data
from ivf_live_birth.model import cross_validate, train_model


def test_training_and_test_auc_above_chance():
    df = generate_cohort(n_cycles=1500, seed=4)
    X_train, y_train, X_test, y_test = prepare_data(df)
    model = train_model(X_train, y_train, config=ExperimentConfig(n_cycles=1500))
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc > 0.65  # synthetic data is separable by construction


def test_cv_returns_sane_values():
    df = generate_cohort(n_cycles=800, seed=4)
    X_train, y_train, _, _ = prepare_data(df)
    cv = cross_validate(X_train, y_train, ExperimentConfig())
    assert cv["cv_folds"] == 5
    assert 0.5 < cv["cv_roc_auc_mean"] <= 1.0


def test_model_exports_decision_function_as_probability():
    df = generate_cohort(n_cycles=400, seed=9)
    X_train, y_train, X_test, _ = prepare_data(df)
    model = train_model(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_reproducible_training():
    df = generate_cohort(n_cycles=600, seed=10)
    X_train, y_train, X_test, y_test = prepare_data(df)
    cfg = ExperimentConfig(n_cycles=600)
    a = train_model(X_train, y_train, config=cfg).predict_proba(X_test)[:, 1]
    b = train_model(X_train, y_train, config=cfg).predict_proba(X_test)[:, 1]
    assert np.allclose(a, b)