"""Calibrated synthetic IVF cohort generator.

Real IVF datasets are clinic-private, so this repository ships a
*biologically informed* simulator instead: every feature is drawn from a
plausible clinical distribution and the live-birth outcome is generated from
a logistic risk model with the well-established signal directions:

* older maternal age          -> lower live-birth odds
* higher AMH / AFC            -> higher odds (ovarian reserve)
* higher FSH                  -> lower odds
* more blastocysts available  -> higher odds
* prior failed cycles         -> lower odds
* male-factor diagnosis       -> sperm metrics matter
* PGT / frozen cycle, day-5 transfer, double transfer  -> small positive effects

Swap in real data anytime: export your cohort as a CSV matching the schema in
``data/README`` and pass ``--csv path/to/data.csv`` to the experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_cohort(
    n_cycles: int | None = None,
    seed: int | None = None,
    config: ExperimentConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    n = n_cycles or cfg.n_cycles

    # --- patient-level covariates -----------------------------------------
    age = np.clip(rng.normal(loc=34.0, scale=4.2, size=n), *cfg.age_range).round(1)

    bmi = np.clip(rng.normal(loc=24.5, scale=4.0, size=n), 17.0, 42.0).round(1)

    # AMH: skewed; declines with age.
    amh = np.clip(
        rng.lognormal(mean=1.1 - 0.055 * (age - 34), sigma=0.45, size=n),
        0.05, 15.0,
    ).round(2)

    # FSH rises as reserve declines; mildly correlated with age.
    fsh = np.clip(
        rng.normal(loc=7.2 + 0.18 * (age - 34) - 0.9 * np.log(amh + 0.5), scale=1.6, size=n),
        3.0, 25.0,
    ).round(1)

    # AFC tracks AMH.
    afc = np.clip(
        rng.normal(loc=6.0 + 4.5 * np.log(amh + 0.5), scale=3.0, size=n),
        1.0, 45.0,
    ).round(0)

    infertility_years = np.clip(rng.exponential(2.2, n), 0.5, 15.0).round(1)

    prior_ivf_cycles = rng.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.30, 0.15, 0.07, 0.03])

    cause = rng.choice(list(cfg.cause_categories), n, p=[0.30, 0.20, 0.12, 0.18, 0.20])

    # Embryos: correlated with reserve and age.
    embryo_rate = np.clip(0.2 * np.log(amh + 0.5) - 0.04 * (age - 34) + 0.8, 0.05, 2.0)
    embryos_blastocyst = rng.poisson(embryo_rate).clip(0, 6)

    transfer_day = rng.choice([3, 5], n, p=[0.35, 0.65])
    num_transferred = rng.choice([1, 2], n, p=[0.60, 0.40])

    is_male_factor = cause == "male_factor"
    sperm_count = np.where(
        is_male_factor,
        np.clip(rng.lognormal(mean=1.6, sigma=0.7, size=n), 0.1, 120.0).round(1),
        rng.uniform(20.0, 120.0, n).round(1),
    )
    sperm_motility = np.where(
        is_male_factor,
        np.clip(rng.normal(loc=42.0, scale=14.0, size=n), 5.0, 90.0).round(1),
        rng.uniform(30.0, 90.0, n).round(1),
    )

    # --- logistic live-birth model ----------------------------------------
    cause_effect = {
        "male_factor": -0.35,
        "tubal": -0.15,
        "endometriosis": -0.45,
        "pcos": 0.15,
        "unexplained": 0.0,
    }
    logits = (
        1.35
        - 0.13 * (age - 34)                    # age penalty
        + 0.85 * np.log(amh + 0.5)              # ovarian reserve
        - 0.16 * (fsh - 7)                      # FSH penalty
        + 0.45 * embryos_blastocyst             # blastocysts available
        - 0.22 * prior_ivf_cycles               # prior failures
        - 0.03 * (bmi - 24.5)                   # mild BMI penalty
        - 0.03 * (infertility_years - 2)        # duration penalty
        + np.array([cause_effect[c] for c in cause])
        + 0.15 * (transfer_day == 5)
        + 0.18 * (num_transferred == 2)
        - 0.45 * is_male_factor                 # male-factor base penalty
        + 0.003 * (sperm_motility - 45)         # ...recovered by good sperm
    )

    p = _sigmoid(logits)
    live_birth = rng.binomial(1, p, n).astype(int)

    # Calibrate the mean predicted probability to the requested population
    # live-birth rate by recentring the logits (probability-scale calibration).
    for _ in range(5):
        mean_p = float(p.mean())
        if abs(mean_p - cfg.target_live_birth_rate) <= 0.005:
            break
        correction = _logit(mean_p) - _logit(cfg.target_live_birth_rate)
        logits = logits - correction
        p = _sigmoid(logits)
    live_birth = rng.binomial(1, p, n).astype(int)

    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": age,
        "bmi": bmi,
        "amh": amh,
        "fsh_day3": fsh,
        "afc": afc,
        "infertility_years": infertility_years,
        "prior_ivf_cycles": prior_ivf_cycles,
        "cause": cause,
        "embryos_blastocyst": embryos_blastocyst,
        "transfer_day": transfer_day,
        "num_transferred": num_transferred,
        "sperm_count_mln": sperm_count,
        "sperm_motility_pct": sperm_motility,
        "live_birth": live_birth,
        # keep the calibrated probability for external validation / honesty
        "true_p_live_birth": p.round(4),
    })

    return df


def _logit(t: float) -> float:
    t = min(max(t, 1e-9), 1 - 1e-9)
    return float(np.log(t / (1 - t)))