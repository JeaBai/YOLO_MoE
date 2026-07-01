#!/usr/bin/env python3
"""Training queue: runs 3 experiments sequentially on VOC 2007.
   1. Experiment: Explicit SparseDualMoE
   2. Baseline A: Standard YOLO26 (no MoE)
   3. Baseline B: Implicit MoE (yolo26_moe)
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

# Common training config
COMMON = {
    'data': 'VOC.yaml',
    'epochs': 200,
    'imgsz': 320,
    'batch': 8,
    'device': 'cpu',
    'workers': 0,
    'exist_ok': True,
}

EXPERIMENTS = [
    {
        'name': 'eamoe_voc',
        'yaml': 'ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml',
        'label': 'Experiment: Explicit SparseDualMoE',
    },
    {
        'name': 'baseline_a_voc',
        'yaml': 'ultralytics/cfg/models/26/yolo26.yaml',
        'label': 'Baseline A: Standard YOLO26 (no MoE)',
    },
    {
        'name': 'baseline_b_voc',
        'yaml': 'ultralytics/cfg/models/moe26/yolo26_moe.yaml',
        'label': 'Baseline B: Implicit MoE (yolo26_moe)',
    },
]


def run_experiment(exp: dict):
    """Run a single experiment. Returns (success, elapsed_seconds, best_mAP50)."""
    label = exp['label']
    name = exp['name']
    yaml_path = exp['yaml']

    print(f"\n{'=' * 60}")
    print(f"  STARTING: {label}")
    print(f"  Config: {yaml_path}")
    print(f"  Save dir: runs/detect/{name}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    try:
        model = YOLO(yaml_path)
        model.train(name=name, **COMMON)
        elapsed = time.time() - t0

        # Extract best mAP50 from results
        best_map50 = 0.0
        save_dir = Path(f'runs/detect/{name}')
        results_csv = save_dir / 'results.csv'
        if results_csv.exists():
            import csv
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    # Strip whitespace from keys
                    first_row = {k.strip(): v for k, v in rows[0].items()}
                    map_key = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in first_row else None
                    for row in rows:
                        row = {k.strip(): v for k, v in row.items()}
                        if map_key and map_key in row:
                            val = float(row[map_key])
                            if val > best_map50:
                                best_map50 = val

        print(f"\n  COMPLETED: {label}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
        print(f"  Best mAP50: {best_map50:.4f}")
        return True, elapsed, best_map50

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED: {label}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False, elapsed, 0.0


def main():
    print("=" * 60)
    print("  TRAINING QUEUE: 3 Experiments on VOC 2007")
    print("  CPU | 320px | batch=8 | 200 epochs each")
    print("=" * 60)

    results = []
    total_start = time.time()

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'─' * 60}")
        print(f"  Queue position: {i + 1}/{len(EXPERIMENTS)}")
        print(f"{'─' * 60}")

        success, elapsed, best_map50 = run_experiment(exp)
        results.append({
            'label': exp['label'],
            'name': exp['name'],
            'success': success,
            'elapsed': elapsed,
            'best_map50': best_map50,
        })

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("  TRAINING QUEUE COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'=' * 60}")
    print(f"\n{'Label':<45} {'Status':<10} {'Best mAP50':<12} {'Time'}")
    print(f"{'─' * 45} {'─' * 10} {'─' * 12} {'─' * 10}")
    for r in results:
        status = 'OK' if r['success'] else 'FAILED'
        time_str = f"{r['elapsed']:.1f}s" if r['elapsed'] < 3600 else f"{r['elapsed']/3600:.1f}h"
        print(f"{r['label']:<45} {status:<10} {r['best_map50']:<12.4f} {time_str}")

    # Check outputs
    print(f"\n  Output files:")
    for r in results:
        if r['success']:
            save_dir = Path(f"runs/detect/{r['name']}")
            for csv_file in ['metrics_core.csv', 'metrics_stability.csv', 'moe_metrics.csv']:
                fpath = save_dir / csv_file
                if fpath.exists():
                    lines = len(open(fpath).readlines()) - 1  # minus header
                    print(f"    {save_dir}/{csv_file}: {lines} rows")


if __name__ == '__main__':
    main()