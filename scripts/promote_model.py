"""Make a trained fold model the one the web app uses (safe model update).

    python scripts/promote_model.py effb0_baseline 0

Steps: copy the candidate into a temporary folder, load it with the SAME code
the app uses, run a test prediction, and only then swap it into MODEL_DIR.
The previous active model is kept as MODEL_DIR.previous for rollback.
Restart the web server afterwards to load the new model.
"""

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.inference import Predictor
from ml.inference.predictor import OPTIONAL_FILES, REQUIRED_FILES


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run_name, fold = sys.argv[1], int(sys.argv[2])
    source = config.ARTIFACTS_DIR / "runs" / run_name / f"fold_{fold}"
    target = config.MODEL_DIR
    staging = target.with_name(target.name + ".staging")
    previous = target.with_name(target.name + ".previous")

    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name in REQUIRED_FILES:
        shutil.copy2(source / name, staging / name)
    for optional in (*OPTIONAL_FILES, "history.json"):
        if (source / optional).exists():
            shutil.copy2(source / optional, staging / optional)

    metadata = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))
    metadata["promoted_at"] = dt.datetime.now().isoformat(timespec="seconds")
    metadata["source_run"] = f"{run_name}/fold_{fold}"
    # Attach the measured pooled held-out results, so the app shows real numbers.
    hybrid_metrics = config.ARTIFACTS_DIR / "runs" / run_name / "evaluation_hybrid_test" / "metrics.json"
    if hybrid_metrics.exists():
        measured = json.loads(hybrid_metrics.read_text(encoding="utf-8"))
        metadata["heldout_performance"] = {
            mode: {"accuracy": round(m["accuracy"], 3), "macro_f1": round(m["macro_f1"], 3),
                   "roc_auc": round(m["roc_auc_macro_ovr"], 3),
                   "osteoporosis_recall": round(m["per_class"]["Osteoporosis"]["recall"], 3), "n": m["n"]}
            for mode, m in measured.items()
        }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")

    # Validate with the production loader before touching the active model.
    predictor = Predictor.from_directory(staging)
    probe = (np.random.default_rng(0).uniform(0, 255, (300, 300))).astype(np.uint8)
    result = predictor.predict(probe, display_size=256)
    assert abs(sum(result.scores.values()) - 1.0) < 1e-4, "scores do not sum to 1"
    if predictor.supports_clinical:
        from ml.data.clinical import encode_clinical

        hybrid = predictor.predict(probe, encode_clinical(60, "female", 1.55, 60, 48), explain=False)
        assert hybrid.mode == "hybrid" and abs(sum(hybrid.scores.values()) - 1.0) < 1e-4
        print(f"Clinical component validated (hybrid mode, osteoporosis boost x{hybrid.osteoporosis_boost}).")
    print(f"Candidate {predictor.model_version} validated (probe output: {result.label}).")

    shutil.rmtree(previous, ignore_errors=True)
    if target.exists():
        target.rename(previous)
    staging.rename(target)
    print(f"Promoted to {target}. Previous model kept at {previous if previous.exists() else '(none)'}.")
    print("Restart the web server to load it.")


if __name__ == "__main__":
    main()
