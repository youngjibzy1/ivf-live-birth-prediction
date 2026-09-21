"""Tests for feature preparation."""

from __future__ import annotations

from ivf_live_birth.data_generator import generate_cohort
from ivf_live_birth.features import prepare_data


def test_split_is_stratified_and_shaped():
    df = generate_cohort(n_cycles=1000, seed=2)
    X_train, y_train, X_test, y_test = prepare_data(df)
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert abs(y_train.mean() - y_test.mean()) < 0.06  # stratification holds


def test_one_hot_encoding_adds_features():
    df = generate_cohort(n_cycles=300, seed=2)
    X_train, _, _, _ = prepare_data(df)
    cause_cols = [c for c in X_train.columns if c.startswith("cause_")]
    assert len(cause_cols) == 4  # 5 causes, drop_first


def test_no_nans_in_features():
    df = generate_cohort(n_cycles=300, seed=2)
    X_train, _, X_test, _ = prepare_data(df)
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()