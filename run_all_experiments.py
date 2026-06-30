#!/usr/bin/env python3
"""Sequential VOC ablation experiments with 50-epoch reporting."""
import sys, os, csv, time
from pathlib import Path
sys.path.insert(0, '/workspace')

# Mock problematic modules
import unittest.mock
for mod in ['matplotlib', 'matplotlib.pyplot', 'seaborn', 'torchvision']:
    sys.modules[mod] = unittest.mock.MagicMock()

with unittest.mock.patch('importlib.metadata.version', return_value='0.0.0'):
    from ultralytics import YOLO

EXPERIMENTS = [
    {
        "name": "baseline_a_yolo26",
        "model": "ultralytics/cfg/models/26/yolo26.yaml",
        "desc": "Baseline A: YOLO26 (no MoE)",
    },
    {
        "name": "baseline_b_implicit_moe",
        "model": "ultralytics/cfg/models/moe26/yolo26_moe.yaml",
        "desc": "Baseline B: YOLO26 + implicit MoE",
    },
    {
        "name": "experiment_explicit_moe",
        "model": "ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml",
        "desc": "Experiment: YOLO26 + explicit MoE (SparseDualMoE)",
    },
]

COMMON_ARGS = {
    "data": "voc.yaml",
    "epochs": 200,
    "imgsz": 320,
    "batch": 8,
    "device": "cpu",
    "workers": 0,
    "exist_ok": True,
    "verbose": False,
}

def report_metrics(exp_name, epoch):
    """Read and report metrics from comprehensive_metrics.csv."""
    csv_path = f"runs/detect/{exp_name}/comprehensive_metrics.csv"
    if not Path(csv_path).exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]  # latest epoch

def print_report(exp_name, desc, epoch):
    row = report_metrics(exp_name, epoch)
    if not row:
        print(f"  [No data yet]")
        return

    mAP50 = float(row.get('mAP50', 0))
    mAP50_95 = float(row.get('mAP50-95', 0))
    box = float(row.get('box_loss', 0))
    cls = float(row.get('cls_loss', 0))
    dfl = float(row.get('dfl_loss', 0))
    topk_p4 = float(row.get('avg_topk_P4', 'nan'))
    topk_p5 = float(row.get('avg_topk_P5', 'nan'))
    desc_p4 = float(row.get('avg_descriptor_P4', 'nan'))
    desc_p5 = float(row.get('avg_descriptor_P5', 'nan'))
    e2_p4 = float(row.get('expert_usage_P4_e2', 'nan'))
    e2_p5 = float(row.get('expert_usage_P5_e2', 'nan'))

    print(f"  mAP50={mAP50:.4f}  mAP50-95={mAP50_95:.4f}  "
          f"box={box:.3f}  cls={cls:.3f}  dfl={dfl:.4f}")
    if not (str(topk_p4) == 'nan'):
        print(f"  MoE: topk P4={topk_p4:.2f} P5={topk_p5:.2f}  "
              f"desc P4={desc_p4:.3f} P5={desc_p5:.3f}  "
              f"e2_usage P4={e2_p4:.3f} P5={e2_p5:.3f}")

def main():
    total_start = time.time()

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i+1}/3: {exp['desc']}")
        print(f"Model: {exp['model']}  |  Output: runs/detect/{exp['name']}")
        print(f"{'='*70}")

        # Check resume
        last_pt = Path(f"runs/detect/{exp['name']}/weights/last.pt")
        resume = last_pt.exists()
        if resume:
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            model = YOLO(exp["model"])

        # Train with progress callback
        exp_start = time.time()
        try:
            model.train(**COMMON_ARGS, name=exp["name"], resume=resume)
        except Exception as e:
            print(f"[ERROR] {exp['desc']}: {e}")
            continue

        elapsed = (time.time() - exp_start) / 3600
        print(f"\n[DONE] {exp['desc']} completed in {elapsed:.1f}h")

        # Final report
        print(f"\n--- Final Metrics for {exp['desc']} (epoch 200) ---")
        print_report(exp['name'], exp['desc'], 200)

    total_elapsed = (time.time() - total_start) / 3600
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_elapsed:.1f}h")
    print(f"Results:")
    for exp in EXPERIMENTS:
        print(f"  {exp['name']}: runs/detect/{exp['name']}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()