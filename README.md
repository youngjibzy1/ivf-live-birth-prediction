# 👶 IVF Live-Birth Prediction

[![CI](https://github.com/youngjibzy1/ivf-live-birth-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/youngjibzy1/ivf-live-birth-prediction/actions)

> Machine-learning prediction of live-birth occurrence **before** IVF treatment — the open-source companion to my [Nature Scientific Reports paper](https://www.nature.com/articles/s41598-020-78937-8) on the same question. Gradient boosting + logistic-regression baseline, strict train/test protocol, cross-validation, clinical operating-point analysis.

Because real IVF cohorts are clinic-private, the repo ships a **biologically-calibrated synthetic cohort generator** with realistic feature relationships (age, AMH, FSH, AFC, blastocysts, prior failures, diagnosis, sperm metrics) and a documented schema so **your own CSV data drops in unchanged**.

## 🚀 Quickstart

```bash
git clone https://github.com/youngjibzy1/ivf-live-birth-prediction.git
cd ivf-live-birth-prediction

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                             # eval suite
python scripts/run_experiment.py --seed 42 --cycles 2500
```

Sample report (seed 42, 2,500 synthetic cycles):

```
IVF live-birth prediction - experiment report
==============================================
Data source        : synthetic-calibrated
Cycles             : 2500  (live-birth rate 32.8%)
Cross-validation   : ROC-AUC 0.753 (+/- 0.020, 5 folds)
Hold-out test set  : n=500
  ROC-AUC          : 0.818
  PR-AUC           : 0.734
  Brier score      : 0.153
  Youden threshold : 0.342 (sens 0.73, spec 0.76)

Top predictors (permutation importance):
  age                      +0.0580
  embryos_blastocyst       +0.0556
  fsh_day3                 +0.0345
  amh                      +0.0249
  infertility_years        +0.0153
```

## 🧬 What's in the box

| Piece | Details |
|---|---|
| **Calibrated data generator** | `data_generator.py` — plausible clinical distributions; live-birth outcome via a logistic risk model with well-established signal directions (age ↓, AMH/AFC ↑, FSH ↓, blastocysts ↑, prior failures ↓, sperm quality ↑ for male factor) |
| **Feature engineering** | `features.py` — one-hot diagnosis, stratified train/test split |
| **Models** | `model.py` — `HistGradientBoostingClassifier` (primary) + standardised `LogisticRegression` (interpretable clinical baseline); 5-fold CV honesty check |
| **Evaluation** | `evaluate.py` — ROC-AUC, PR-AUC, Brier score, Youden operating point, **specificity-fixed (60%) clinical threshold**, permutation importance, ROC/PR/calibration plots |
| **Experiment CLI** | One command: generated data → CV → model → report → model_metrics.png + report.json + per-patient predictions |

## 📦 Data schema (for your own CSV)

| Column | Meaning | Example |
|---|---|---|
| `age` | maternal age at cycle start | 34.2 |
| `bmi` | body mass index | 24.1 |
| `amh` | anti-Müllerian hormone (ng/mL) | 2.4 |
| `fsh_day3` | day-3 FSH (IU/L) | 7.5 |
| `afc` | antral follicle count | 12 |
| `infertility_years` | years trying to conceive | 3 |
| `prior_ivf_cycles` | previous IVF attempts | 1 |
| `cause` | `male_factor/tubal/endometriosis/pcos/unexplained` | male_factor |
| `embryos_blastocyst` | usable blastocysts available | 3 |
| `transfer_day` | day of transfer (3 or 5) | 5 |
| `num_transferred` | embryos transferred | 2 |
| `sperm_count_mln`, `sperm_motility_pct` | sperm metrics (esp. male factor) | 28, 45 |
| `live_birth` | outcome (0/1, prediction target) | 1 |

```bash
python scripts/run_experiment.py --csv path/to/cohort.csv
```

## 📁 Structure

```
src/ivf_live_birth/
  config.py          # experiment + model parameters
  data_generator.py  # calibrated synthetic cohort
  features.py        # encoding + stratified split
  model.py           # HGB + logistic baseline, 5-fold CV
  evaluate.py        # clinical metrics, Youden, calibration, plots
  experiment.py      # end-to-end runner
scripts/run_experiment.py
tests/               # evals (pytest)
```

## ⚠️ Honest limitations

* The synthetic cohort is **calibrated, not real** — feature-effect directions mirror established reproductive medicine, but real performance requires clinical data.
* `score` calibration uses test-set probabilities only; in production you would isotonic/Platt-calibrate on a validation cut.
* Prediction supports shared decision-making pre-treatment; it is not a substitute for medical advice.

## License

MIT — research reference or portfolio piece. For clinical deployment, validate against your own IRB-approved cohort.