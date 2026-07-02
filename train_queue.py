#!/usr/bin/env python3
"""Training queue: Explicit SparseDualMoE → Baseline A (YOLO26) → Baseline B (Implicit MoE)"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# Common training config
TRAIN_CONFIG = {
    'data': 'voc.yaml',
    'epochs': 200,
    'imgsz': 320,
    'batch': 8,
    'device': 'cpu',
    'workers': 0,
    'exist_ok': True,
}

# Experiment queue
EXPERIMENTS = [
    {
        'name': 'voc_explicit_moe',
        'model_yaml': 'ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml',
        'desc': 'Explicit Allocation SparseDualMoE',
    },
    {
        'name': 'voc_baseline_a_yolo26',
        'model_yaml': 'ultralytics/cfg/models/26/yolo26.yaml',
        'desc': 'Baseline A: Standard YOLO26 (no MoE)',
    },
    {
        'name': 'voc_baseline_b_implicit_moe',
        'model_yaml': 'ultralytics/cfg/models/moe26/yolo26_moe.yaml',
        'desc': 'Baseline B: Implicit MoE (UltimateOptimizedMoE)',
    },
]

def run_experiment(idx, exp):
    """Run a single experiment with error handling."""
    print(f"\n{'=' * 60}")
    print(f"Experiment {idx + 1}/{len(EXPERIMENTS)}: {exp['desc']}")
    print(f"Model: {exp['model_yaml']}")
    print(f"Output: runs/detect/{exp['name']}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")
    
    try:
        model = YOLO(exp['model_yaml'])
        results = model.train(
            **TRAIN_CONFIG,
            name=exp['name'],
        )
        print(f"\nExperiment {exp['name']} COMPLETED successfully.\n")
        return True
    except Exception as e:
        print(f"\nExperiment {exp['name']} FAILED: {e}\n")
        traceback.print_exc()
        return False

def main():
    print(f"\nTraining Queue Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Config: epochs={TRAIN_CONFIG['epochs']}, imgsz={TRAIN_CONFIG['imgsz']}, "
          f"batch={TRAIN_CONFIG['batch']}, device={TRAIN_CONFIG['device']}")
    
    results = []
    for idx, exp in enumerate(EXPERIMENTS):
        success = run_experiment(idx, exp)
        results.append((exp['name'], success))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING QUEUE COMPLETED")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    for name, success in results:
        status = 'PASS' if success else 'FAIL'
        print(f"  [{status}] {name}")
    
    # Check output files
    print(f"\nOutput files:")
    for name, success in results:
        csv_path = Path(f'runs/detect/{name}/comprehensive_metrics.csv')
        if csv_path.exists():
            n_lines = len(csv_path.read_text().strip().split('\n')) - 1  # minus header
            print(f"  {name}: comprehensive_metrics.csv ({n_lines} epochs)")
        else:
            print(f"  {name}: comprehensive_metrics.csv NOT FOUND")

if __name__ == '__main__':
    main()