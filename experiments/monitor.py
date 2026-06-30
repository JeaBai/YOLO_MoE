#!/usr/bin/env python3
"""Real-time monitoring script for YOLO MoE ablation experiments.

Reads comprehensive_metrics.csv from each experiment and displays
a comparison table of key metrics for the latest epoch.
"""

import csv
import os
import time
from pathlib import Path

EXPERIMENT_DIRS = [
    ("Baseline A (no MoE)", "runs/detect/baseline_a_yolo26"),
    ("Baseline B (implicit MoE)", "runs/detect/baseline_b_implicit_moe"),
    ("Experiment (explicit MoE)", "runs/detect/experiment_explicit_moe"),
]


def read_last_row(csv_path):
    """Read the last row of a CSV file."""
    if not Path(csv_path).exists():
        return None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-1] if rows else None


def display_table():
    """Display comparison table of key metrics."""
    headers = [
        "Experiment",
        "Epoch",
        "mAP50",
        "mAP50-95",
        "box_loss",
        "cls_loss",
        "avg_descriptor_P4",
        "avg_descriptor_P5",
        "avg_topk_P4",
        "avg_topk_P5",
        "e2_usage_P4",
        "e2_usage_P5",
        "dead_P4",
        "dead_P5",
    ]

    rows = []
    for label, exp_dir in EXPERIMENT_DIRS:
        csv_path = os.path.join(exp_dir, "comprehensive_metrics.csv")
        row = read_last_row(csv_path)
        if row is None:
            rows.append([label, "N/A"] + ["-"] * (len(headers) - 2))
            continue

        # Compute dead experts from usage
        dead_p4 = 0
        dead_p5 = 0
        for ei in range(4):
            if float(row.get(f"expert_usage_P4_e{ei}", 1.0)) < 0.01:
                dead_p4 += 1
            if float(row.get(f"expert_usage_P5_e{ei}", 1.0)) < 0.01:
                dead_p5 += 1

        rows.append([
            label,
            row.get("epoch", "?"),
            f"{float(row.get('mAP50', 0)):.4f}",
            f"{float(row.get('mAP50-95', 0)):.4f}",
            f"{float(row.get('box_loss', 0)):.4f}",
            f"{float(row.get('cls_loss', 0)):.4f}",
            f"{float(row.get('avg_descriptor_P4', 0)):.3f}",
            f"{float(row.get('avg_descriptor_P5', 0)):.3f}",
            f"{float(row.get('avg_topk_P4', 0)):.2f}",
            f"{float(row.get('avg_topk_P5', 0)):.2f}",
            f"{float(row.get('expert_usage_P4_e2', 0)):.3f}",
            f"{float(row.get('expert_usage_P5_e2', 0)):.3f}",
            str(dead_p4),
            str(dead_p5),
        ])

    # Print table
    col_widths = [max(len(str(r[i])) for r in rows + [headers]) + 2 for i in range(len(headers))]

    # Header
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-" * len(header_line)

    print("\n" + "=" * len(header_line))
    print("MoE Ablation Experiment Monitor")
    print("=" * len(header_line))
    print(header_line)
    print(sep_line)

    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

    print(sep_line)


def watch(interval=30):
    """Continuously monitor experiments."""
    print(f"Monitoring every {interval}s. Press Ctrl+C to stop.")
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            display_table()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor MoE ablation experiments")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor (refresh every 30s)")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    args = parser.parse_args()

    if args.watch:
        watch(args.interval)
    else:
        display_table()