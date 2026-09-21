"""Tests for the calibrated cohort generator."""

from __future__ import annotations

import numpy as np

from ivf_live_birth.config import ExperimentConfig
from ivf_live_birth.data_generator import generate_cohort


def test_cohort_shape_and_no_missing():
    df = generate_cohort(n_cycles=500, seed=1)
    assert len(df) == 500
    assert df.isna().sum().sum() == 0


def test_live_birth_rate_near_target():
    df = generate_cohort(n_cycles=5000, seed=3, config=ExperimentConfig())
    assert 0.25 < df.live_birth.mean() < 0.40


def test_age_effect_is_negative():
    """Older patients must have systematically lower mean live-birth rate."""
    df = generate_cohort(n_cycles=8000, seed=5)
    young = df[df.age < 33].live_birth.mean()
    old = df[df.age >= 40].live_birth.mean()
    assert young > old


def test_reserve_markers_are_positive():
    """Higher AMH / more blastocysts must associate with higher birth rate."""
    df = generate_cohort(n_cycles=8000, seed=6)
    high_amh = df[df.amh > 3.0].live_birth.mean()
    low_amh = df[df.amh < 0.8].live_birth.mean()
    assert high_amh > low_amh

    many = df[df.embryos_blastocyst >= 3].live_birth.mean()
    none = df[df.embryos_blastocyst == 0].live_birth.mean()
    assert many > none


def test_reproducible_with_seed():
    a = generate_cohort(n_cycles=200, seed=11)
    b = generate_cohort(n_cycles=200, seed=11)
    assert a.equals(b)


def test_all_cause_categories_present():
    df = generate_cohort(n_cycles=2000, seed=8)
    assert set(df.cause.unique()) == set(ExperimentConfig().cause_categories)