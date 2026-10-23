"""Measure real CPU inference latency of the active model on this machine.

    python scripts/benchmark_latency.py [--n 20]

Times three stages per image: decode+preprocess, model forward pass, and
Grad-CAM + overlay. The first call is reported separately (graph warm-up).
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from ml.data.manifest import load_manifest
from ml.inference import Predictor
from ml.preprocessing import decode_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    started = time.perf_counter()
    predictor = Predictor.from_directory(config.MODEL_DIR)
    load_seconds = time.perf_counter() - started

    records = [r for r in load_manifest(config.DATA_ARTIFACTS_DIR / "manifest.csv") if not r.excluded]
    files = [(config.DATASET_DIR / r.path).read_bytes() for r in records[: args.n + 1]]

    totals, forward_only = [], []
    first = None
    for i, data in enumerate(files):
        t0 = time.perf_counter()
        image = decode_image(data)
        predictor.predict(image, explain=True)
        elapsed = time.perf_counter() - t0
        if i == 0:
            first = elapsed
            continue
        totals.append(elapsed)
        t1 = time.perf_counter()
        predictor.predict(image, explain=False)
        forward_only.append(time.perf_counter() - t1)

    result = {
        "machine": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "model_version": predictor.model_version,
        "model_load_seconds": round(load_seconds, 2),
        "first_request_seconds": round(first, 3),
        "n_timed": len(totals),
        "full_pipeline_median_s": round(statistics.median(totals), 3),
        "full_pipeline_p95_s": round(float(np.percentile(totals, 95)), 3),
        "without_gradcam_median_s": round(statistics.median(forward_only), 3),
    }
    out = config.ARTIFACTS_DIR / "benchmark" / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key, value in result.items():
        print(f"{key:<28}{value}")


if __name__ == "__main__":
    main()
