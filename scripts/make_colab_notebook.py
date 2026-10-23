"""Generate notebooks/colab_xray_experiments.ipynb and the code zip to upload with it.

    python scripts/make_colab_notebook.py

Outputs (outside the git repo, on D:):
    D:/Projects/Oestoporosis/colab_package/osteo_code.zip   project code only (git-tracked files)
and notebooks/colab_xray_experiments.ipynb in the repo.
"""

import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT.parent / "colab_package"
FOLDS_FINGERPRINT = "b197b36dcab373ce"  # sha256 of artifacts/data/folds.json "parts" on the laptop
# Same versions as the laptop, except TensorFlow on Python 3.13 (2.19 has no 3.13 build; 2.20
# accepts the same Keras 3.10.0 and NumPy 2.1.3). Results here are metrics only; models stay in Colab.
PINNED = ("keras==3.10.0 numpy==2.1.3 opencv-python-headless==4.14.0.94 "
          "scikit-learn==1.9.1 scipy==1.18.1 h5py==3.16.0 matplotlib==3.11.2 pillow==13.0.0 "
          "python-dotenv==1.2.4 openpyxl==3.1.5")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# X-ray-only experiments on a Colab GPU

Trains EfficientNet-B0 with the project's own code on 4 input variants x 2 tasks, 5 folds each,
and reports **validation** results only (the test folds are not touched).

| Variant | Input |
|---|---|
| whole224 | whole picture, 224 px (current baseline) |
| crop224 | black borders removed, 224 px |
| knees224 | two-knee pictures split into single knees, 224 px |
| knees384 | single knees at 384 px |

Tasks: **3class** (Normal/Osteopenia/Osteoporosis) and **binary** (Normal vs Osteoporosis).

**Before you start:** Runtime → Change runtime type → **T4 GPU** or a **TPU** → Save.
Then drag `osteo_code.zip` (from `D:/Projects/Oestoporosis/colab_package/`) into the **Files** panel (📁 on the left)
and choose **Runtime → Run all**. The images are downloaded by Colab itself from Mendeley, so nothing large is uploaded.
""")

md("""
## 1. Detect the accelerator (T4 GPU or TPU) and Python version

* **GPU (T4):** runs the project's TensorFlow code as on the laptop.
* **TPU (e.g. v5e):** Keras runs on its **JAX** engine instead (TensorFlow is only used for the data pipeline).
  The TPU is checked from a short helper process, because only one process may hold a TPU at a time.
""")
code("""
import os, sys, shutil, subprocess
print("Python", sys.version.split()[0])
assert sys.version_info[:2] >= (3, 10), "Python too old"
if shutil.which("nvidia-smi"):
    ACCEL = "gpu"
    print("GPU:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip())
else:
    probe = subprocess.run([sys.executable, "-c", "import jax; d = jax.devices(); print(d[0].platform, len(d), d[0].device_kind)"],
                           capture_output=True, text=True)
    print("JAX probe:", probe.stdout.strip() or probe.stderr.strip()[-300:])
    if probe.stdout.startswith("tpu"):
        ACCEL = "tpu"
    else:
        raise SystemExit("No GPU or TPU attached. Runtime -> Change runtime type -> T4 GPU or a TPU -> Save, then Run all.")
os.environ["KERAS_BACKEND"] = "jax" if ACCEL == "tpu" else "tensorflow"   # inherited by every !python below
print("Accelerator:", ACCEL, "| Keras backend:", os.environ["KERAS_BACKEND"])
""")

md("## 2. Unpack the project code (uploaded zip, ~1 MB)")
code("""
import zipfile
from pathlib import Path
from google.colab import files
zips = [p for p in Path("/content").glob("*.zip") if "osteo_code" in p.name]
if not zips:
    uploaded = files.upload()                       # choose osteo_code.zip
    zips = [Path("/content") / n for n in uploaded]
PROJECT = Path("/content/osteo")
with zipfile.ZipFile(zips[0]) as z:
    z.extractall(PROJECT)
print(sorted(p.name for p in PROJECT.iterdir()))
""")

md("""
## 3. Install libraries (~2-3 minutes)

* **GPU:** same versions as the laptop (TensorFlow 2.20 instead of 2.19 on Python 3.13, which has no 2.19 build).
* **TPU:** keeps Colab's own JAX/TPU installation untouched, adds Keras 3.15.1 (needed for current JAX)
  and CPU-only TensorFlow for the data pipeline.

Only metrics come back from Colab, so these versions do not affect the web app.
""")
code(f"""
import sys
COMMON = "opencv-python-headless==4.14.0.94 scikit-learn==1.9.1 python-dotenv==1.2.4 openpyxl==3.1.5 matplotlib pillow"
if ACCEL == "gpu":
    tf_version = "2.19.0" if sys.version_info[:2] <= (3, 12) else "2.20.0"
    !pip -q install "tensorflow[and-cuda]=={{tf_version}}" keras==3.10.0 numpy==2.1.3 h5py {{COMMON}}
else:
    !pip -q install keras==3.15.1 tensorflow-cpu==2.20.0 h5py {{COMMON}}
!python -c "import keras; print('Keras', keras.__version__, '| backend', keras.backend.backend())"
if ACCEL == "tpu":
    !python -c "import jax; print('JAX', jax.__version__, '| devices:', jax.devices())"
else:
    !python -c "import tensorflow as tf; print('TF', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"
""")

md("## 4. Download the Mendeley dataset (public, checksum-verified) and configure paths")
code("""
import subprocess, zipfile
raw = PROJECT / "data" / "raw"; raw.mkdir(parents=True, exist_ok=True)
if not (raw / "Osteoporosis Knee X-ray").exists():
    url = "https://data.mendeley.com/public-files/datasets/fxjm8fb6mw/files/b7a95139-dfef-4db3-b3b9-e0a36315f334/file_downloaded"
    subprocess.run(["curl", "-sL", "-o", "/content/mendeley.zip", url], check=True)
    digest = subprocess.run(["sha256sum", "/content/mendeley.zip"], capture_output=True, text=True).stdout.split()[0]
    assert digest == "2d4d3a139f30c6b89eb123791af54db8c78ea143b3cfc98c3431665e54ace8b3", "Mendeley checksum mismatch"
    with zipfile.ZipFile("/content/mendeley.zip") as z:
        z.extractall(raw)
(PROJECT / ".env").write_text("DATASET_DIR=data/raw/Osteoporosis Knee X-ray\\n")
print(sorted(p.name for p in (raw / "Osteoporosis Knee X-ray").iterdir()))
""")

md("""
## 5. Rebuild the cleaned manifest and folds, and check they match the laptop

The fold fingerprint must equal the laptop's, which proves the same patients sit in the same folds.
""")
code(f"""
import json, hashlib
!cd /content/osteo && python scripts/prepare_dataset.py | head -8
parts = json.loads((PROJECT / "artifacts/data/folds.json").read_text())["parts"]
fp = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
print("folds fingerprint:", fp)
assert fp == "{FOLDS_FINGERPRINT}", "Folds differ from the laptop - stop here and do not train (check the dataset download)"
print("OK - identical folds to the laptop")
""")

md("""
## 6. Run the experiments (roughly 45-90 minutes on a T4; TPU timing unknown - first epochs compile slowly)

Progress is printed per fold. Results are saved after every variant, so a disconnect loses at most one.
If the session disconnects: reconnect, run cells 2-5 again, then run this cell with
`--variants` limited to the ones not yet finished (see `summary.json`).
""")
code("""
!cd /content/osteo && TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 python scripts/experiment_xray.py 2>&1 | grep -vE "NodeDef|oneDNN|absl|cpu_feature|retracing|UserWarning|data_adapter|^Epoch|step - "
""")

md("## 7. Results: copy this table and send it back")
code("""
print((PROJECT / "artifacts/experiments/xray/summary.md").read_text())
""")

md("## 8. Download the results (small JSON + markdown)")
code("""
import shutil
from google.colab import files
out = shutil.make_archive("/content/xray_experiment_results", "zip", PROJECT / "artifacts/experiments/xray")
files.download(out)
""")


def build_code_zip() -> Path:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    target = PACKAGE_DIR / "osteo_code.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in listed:
            path = ROOT / rel
            if path.is_file() and not rel.endswith(".ipynb"):
                z.write(path, rel)
    return target


if __name__ == "__main__":
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    nb_path = ROOT / "notebooks" / "colab_xray_experiments.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    zip_path = build_code_zip()
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    print(f"notebook: {nb_path}")
    print(f"code zip: {zip_path}  ({zip_path.stat().st_size / 1e6:.2f} MB, {len(names)} files)")
    forbidden = [n for n in names if n.startswith(("data/", "artifacts/", "uploads/", "reports/", "instance/", ".venv/"))
                 or n == ".env" or n.endswith((".keras", ".db"))]
    print("forbidden files in zip:", forbidden or "none")
