"""Configuration for the IVF live-birth prediction experiment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_cycles: int = 2500          # number of IVF cycles to simulate
    test_size: float = 0.20
    cv_folds: int = 5

    # Targets for the calibrated synthetic generator (approx population stats).
    target_live_birth_rate: float = 0.32
    age_range: tuple[float, float] = (22.0, 45.0)

    # Model parameters (HistGradientBoosting wins on tabular data).
    hgb_max_iter: int = 250
    hgb_learning_rate: float = 0.08
    hgb_max_depth: int = 4

    # Feature columns (order matters for consistency).
    feature_cols: tuple[str, ...] = (
        "age", "bmi", "amh", "fsh_day3", "afc", "infertility_years",
        "prior_ivf_cycles", "embryos_blastocyst", "transfer_day",
        "num_transferred", "sperm_count_mln", "sperm_motility_pct",
    )

    cause_categories: tuple[str, ...] = (
        "male_factor", "tubal", "endometriosis", "pcos", "unexplained",
    )

    outcome_col: str = "live_birth"

    output_dir: str = "output"