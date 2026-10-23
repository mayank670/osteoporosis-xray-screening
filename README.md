# Osteoporosis Detection Using Deep Learning from Knee X-ray Images

An **experimental academic screening-support prototype**. It classifies a knee
X-ray (optionally with patient details) as *Normal*, *Osteopenia* or *Osteoporosis*, shows a Grad-CAM heatmap
and produces a PDF report. It includes a Flask web app with accounts, scan
history and an admin page.

> **Not a medical device.** This software does not diagnose osteoporosis,
> does not replace a DXA scan, and does not replace a clinician. Model outputs
> are uncalibrated scores, not probabilities of disease. Measured performance
> on the available data is **modest**: see [Results](#results).

## Project layout

```
app/            Flask web app: routes/ (thin), services/ (logic), templates/, static/
ml/             No Flask imports: preprocessing/, data/, models/, training/,
                evaluation/, explainability/, inference/
scripts/        Command-line entry points (prepare, train, evaluate, benchmark, promote)
tests/          pytest suite (synthetic fixtures only)
notebooks/      Optional Google Colab notebooks (dataset audit, GPU experiments)
config.py       Shared configuration (class order, preprocessing, training hyperparameters)
run.py          Development server          serve.py   Production server (Waitress)
data/ artifacts/ uploads/ reports/ instance/   Generated or private — git-ignored
```

## 1. Setup (Windows, PowerShell)

Requirements: **Python 3.12** (TensorFlow 2.19 has no wheels for 3.13+), about
3 GB of disk space and 6 GB of RAM. A GPU is not needed.

```powershell
py install 3.12                       # Python install manager; or get 3.12 from python.org
py -V:3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env                # then edit .env
python scripts\check_environment.py   # prints versions; inspects EfficientNet-B0
```

On Linux or macOS: `python3.12 -m venv .venv && source .venv/bin/activate`;
everything else is the same.

## 2. Dataset

1. Download the **Knee X-ray Osteoporosis Database**: Wani & Arora (2021),
   Mendeley Data, [doi:10.17632/fxjm8fb6mw.2](https://doi.org/10.17632/fxjm8fb6mw.2),
   licence CC BY 4.0.
2. Check the zip's SHA-256: `2d4d3a139f30c6b89eb123791af54db8c78ea143b3cfc98c3431665e54ace8b3`.
3. Extract it to `data/raw/`.
4. Set `DATASET_DIR=data/raw/Osteoporosis Knee X-ray` in `.env`.
5. Run `python scripts\prepare_dataset.py`. This writes the cleaned manifest,
   which excludes 29 duplicate or conflicting files and leaves 210 usable
   images. It also writes the 5 leakage-checked folds.

Optional: `python scripts\visualize_preprocessing.py` saves
`artifacts/figures/preprocessing_steps.png`.

## 3. Training (optional for running the app)

```powershell
python scripts\train.py --run-name effb0_baseline            # 5 folds, about 45 min on a 4-thread CPU
python scripts\evaluate.py --run-name effb0_baseline --subset val
python scripts\benchmark_backbones.py                        # 4 backbones + CLAHE, validation only
python scripts\explore_improvements.py                       # resolution / texture / clinical search, validation only
python scripts\train_clinical.py --run-name effb0_baseline   # clinical model per fold (seconds)
python scripts\evaluate_hybrid.py --run-name effb0_baseline --subset val
```

Each fold writes to `artifacts/runs/<run>/fold_<k>/`: `model.keras`,
`class_map.json`, `preprocessing.json`, `metadata.json`, `history.json` and
`training_log.csv`. The run folder also gets copies of the manifest and the
fold assignment.

## 4. Evaluation on the held-out test folds (once)

```powershell
python scripts\evaluate.py --run-name effb0_baseline --subset test
python scripts\evaluate_hybrid.py --run-name effb0_baseline --subset test
```

This writes `evaluation_test/` (`report.md`, `metrics.json`,
`predictions.csv`, confusion matrix, ROC and training curves). Each run is
logged in `artifacts/test_evaluations.log`. Do not tune anything after
looking at this output.

## 5. Install a model for the web app

```powershell
python scripts\promote_model.py effb0_baseline 0    # validates, then installs to artifacts/models/active
python scripts\benchmark_latency.py                 # measures CPU latency on this machine
```

The fold is predefined (fold 0). It was **not** chosen by test score, because
choosing by test score would turn the test set into a selection set. If no
model is installed, the app still starts, `/health` reports
`model_loaded: false`, and the upload page explains how to install one. The
app never substitutes an untrained model.

## 6. Run the web app

```powershell
python run.py                                   # development: http://127.0.0.1:5000
flask --app run create-admin                     # optional: create an administrator
```

To make yourself an admin, register normally first, then run `create-admin`
with the same email.

## 7. Production deployment

The Flask development server (`run.py`) is **not** for production.

```powershell
$env:APP_ENV = "production"
$env:SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
python serve.py                                  # Waitress on 127.0.0.1:8000
```

- Put it behind an HTTPS reverse proxy (IIS ARR, Caddy or nginx).
  Production cookies are marked `Secure`, so logins need HTTPS.
- Keep `uploads/`, `reports/`, `instance/` and `artifacts/` outside any
  public web root. The app serves them only through authenticated,
  ownership-checked routes.
- Retention: schedule `flask --app run purge-scans` (Windows Task
  Scheduler). It deletes scans older than `RETENTION_DAYS` (default 90),
  together with their files.
- Logs are written to `instance/logs/app.log`, rotated at 1 MB with 5 kept.

### Database setup and backup

- The schema is created automatically on first start (`db.create_all()`), in
  SQLite at `instance/app.db`. Set `DATABASE_URL` to use another database.
- **Backup:** stop the server (or use SQLite's online backup), then copy
  `instance/app.db`, `uploads/` and `reports/` *together*. The database
  stores paths into those folders.
  ```powershell
  sqlite3 instance\app.db ".backup backups\app-$(Get-Date -f yyyyMMdd).db"
  ```
- **Restore:** stop the server, then put back the database file and the two
  folders from the same backup.

## 8. Tests

```powershell
python -m pytest
```

The suite has 143 tests and takes about 90 seconds on a 4-thread CPU, partly because
it builds real backbone graphs. All tests use synthetic images and a tiny
untrained model. They check software behaviour, **not** medical performance.

## Results

These are pooled held-out test predictions from 5-fold cross-validation
(n = 210), with 95% confidence intervals. Labels are QUS T-score categories.
The app offers two modes:

| Metric | X-ray only | **X-ray + patient details (deployed, with screening rule)** | Design target |
|---|---|---|---|
| Accuracy | 0.538 (0.47-0.60) | **0.638** (0.57-0.70) | >= 0.90 |
| Macro F1 | 0.367 (0.30-0.43) | **0.632** (0.56-0.69) | >= 0.88 |
| Osteoporosis recall | 0.209 (0.11-0.35) | **0.767** (0.62-0.87) | >= 0.90 |
| ROC-AUC (macro OvR) | 0.555 | **0.817** | - |
| CPU latency incl. Grad-CAM | median 1.58 s, p95 2.97 s | same | < 3 s |

- **X-ray only is close to chance** on this dataset: 210 JPEG images, QUS
  labels, one site, label conflicts. Higher resolution, tiles and texture
  features did not help (validation-only search).
- **Adding patient details nearly doubles macro F1.** Age, sex, height,
  weight, menopause and risk factors are fused with the CNN; the T-score is
  never used.
- **Screening rule:** the label is Osteoporosis whenever 1.2 x its score is
  the highest (chosen on validation). This catches more cases (77% instead of
  65%) at the cost of more false alarms among Osteopenia patients.
- Patient details are **ticked by default** on the upload page.
- Most of the hybrid model's signal comes from those details. The accuracy
  targets are **still not met**, so this is a research prototype, not a
  screening tool.

## Licence and attribution

The dataset is © Wani & Arora, CC BY 4.0, and must be cited:
Wani, I. M., & Arora, S. (2021). *Knee X-ray Osteoporosis Database* (Version 2)
[Data set]. Mendeley Data. https://doi.org/10.17632/fxjm8fb6mw.2.
The authors ask that it be used for research only. Pretrained ImageNet weights
come from `keras.applications`.
