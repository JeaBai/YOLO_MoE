#!/usr/bin/env python3
"""VOC-scale training queue for MoE ablation experiments.

Runs three experiments sequentially:
  1. Baseline A: YOLO26 (no MoE)
  2. Baseline B: YOLO26 + implicit MoE (yolo26_moe)
  3. Experiment: YOLO26 + explicit MoE (SparseDualMoE)

Each experiment records comprehensive_metrics.csv and stability_metrics.csv.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

EXPERIMENTS = [
    {
        "name": "baseline_a_yolo26",
        "model": "ultralytics/cfg/models/26/yolo26.yaml",
        "description": "Baseline A: Standard YOLO26 (no MoE)",
        "pretrained": None,  # no MoE pretrained available
    },
    {
        "name": "baseline_b_implicit_moe",
        "model": "ultralytics/cfg/models/moe26/yolo26_moe.yaml",
        "description": "Baseline B: YOLO26 + implicit MoE (black-box router)",
        "pretrained": None,
    },
    {
        "name": "experiment_explicit_moe",
        "model": "ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml",
        "description": "Experiment: YOLO26 + explicit MoE (deterministic descriptor)",
        "pretrained": None,
    },
]

COMMON_ARGS = {
    "data": "voc.yaml",
    "epochs": 200,
    "imgsz": 640,
    "batch": 16,
    "device": "cpu",
    "workers": 0,
    "exist_ok": True,
    "verbose": False,
}


def main():
    parser = argparse.ArgumentParser(description="Run MoE ablation experiments on VOC")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
    parser.add_argument("--skip", type=int, nargs="+", default=[], help="Skip experiment indices (0, 1, 2)")
    parser.add_argument("--only", type=int, nargs="+", default=None, help="Only run these experiment indices")
    args = parser.parse_args()

    for i, exp in enumerate(EXPERIMENTS):
        if args.only is not None and i not in args.only:
            continue
        if i in args.skip:
            print(f"\n[SKIP] Experiment {i}: {exp['description']}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Experiment {i + 1}/3: {exp['description']}")
        print(f"Model: {exp['model']}")
        print(f"Output: runs/detect/{exp['name']}")
        print(f"{'=' * 60}")

        # Check if we should resume
        last_pt = Path(f"runs/detect/{exp['name']}/weights/last.pt")
        resume = args.resume and last_pt.exists()

        if resume:
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            model = YOLO(exp["model"])
            if exp["pretrained"] and Path(exp["pretrained"]).exists():
                print(f"Loading pretrained weights: {exp['pretrained']}")
                # Will be handled by YOLO internally

        try:
            results = model.train(
                **COMMON_ARGS,
                name=exp["name"],
                resume=resume,
            )
            print(f"[DONE] {exp['description']} completed.")
        except Exception as e:
            print(f"[ERROR] {exp['description']} failed: {e}")
            # Continue with next experiment
            continue

    print(f"\n{'=' * 60}")
    print("All experiments completed.")
    print("Results:")
    for exp in EXPERIMENTS:
        print(f"  {exp['name']}: runs/detect/{exp['name']}/")


if __name__ == "__main__":
    main()