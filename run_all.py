#!/usr/bin/env python3
"""
PHM Foundation Model — End-to-End Pipeline
============================================

Self-supervised masked autoencoder with frequency-conditioned
normalization for multi-domain fault diagnosis & prognosis.

Pipeline Steps:
  1. Load & preprocess real PHM datasets (CWRU, PRONOSTIA, CMAPSS, MFPT, UOC18)
  2. Train per-dataset baseline CNNs
  3. Self-supervised pre-training (masked autoencoder)
  4. Three-stage fine-tuning per dataset
  5. Comprehensive evaluation (comparison, few-shot, cross-domain, t-SNE)
  6. Ablation studies
  7. Summary report

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-ablations   # Skip ablations (faster)
    python run_all.py --step 3           # Run only step 3
    python run_all.py --quick            # Quick mode (fewer epochs)
"""

import sys
import os
import argparse
import time

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, set_seed, ensure_dirs


# ═════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═════════════════════════════════════════════════════════════════════

def step1_data():
    """Load real PHM datasets via phmd, preprocess, store in HDF5."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PIPELINE — Loading Real PHM Datasets")
    print("=" * 70)
    from data_pipeline import generate_all_datasets, verify_datasets
    generate_all_datasets()
    verify_datasets()


def step2_baseline():
    """Train per-dataset baseline CNNs for comparison."""
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE CNN TRAINING")
    print("=" * 70)
    from train_baseline import train_baseline
    train_baseline()


def step3_pretrain():
    """Self-supervised masked autoencoder pre-training."""
    print("\n" + "=" * 70)
    print("STEP 3: SELF-SUPERVISED PRE-TRAINING (Masked Autoencoder)")
    print("=" * 70)
    from pretrain import pretrain
    pretrain()


def step4_finetune():
    """Three-stage fine-tuning of pre-trained encoder."""
    print("\n" + "=" * 70)
    print("STEP 4: THREE-STAGE FINE-TUNING")
    print("=" * 70)
    from fine_tune import fine_tune
    fine_tune()


def step5_evaluation():
    """Comprehensive evaluation suite."""
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATION (comparison, few-shot, cross-domain, t-SNE)")
    print("=" * 70)
    from evaluation import run_evaluation
    run_evaluation()


def step6_ablations():
    """Ablation studies for component analysis."""
    print("\n" + "=" * 70)
    print("STEP 6: ABLATION STUDIES")
    print("=" * 70)
    from ablation_studies import run_ablations
    run_ablations()


def step7_summary():
    """Generate summary report from all results."""
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    import pandas as pd

    report = []

    # ── Comparison ───────────────────────────────────────────────────
    if os.path.exists("results/comparison_table.csv"):
        df = pd.read_csv("results/comparison_table.csv")
        report.append("=" * 70)
        report.append("PERFORMANCE COMPARISON: Baseline CNN vs Foundation Model")
        report.append("=" * 70)
        report.append(df.to_string(index=False))

        # Summary statistics
        for metric_bl, metric_fd, label, higher_better in [
            ("Baseline Accuracy", "Foundation Accuracy", "Accuracy", True),
            ("Baseline F1 Score", "Foundation F1 Score", "F1 Score", True),
            ("Baseline Rul Mae", "Foundation Rul Mae", "RUL MAE", False),
        ]:
            if metric_bl in df.columns and metric_fd in df.columns:
                valid = df[df[metric_bl] > 0]
                if not valid.empty:
                    avg_bl = valid[metric_bl].mean()
                    avg_fd = valid[metric_fd].mean()
                    report.append(f"\n  Avg Baseline {label}: {avg_bl:.4f}")
                    report.append(f"  Avg Foundation {label}: {avg_fd:.4f}")
                    diff = avg_fd - avg_bl if higher_better else avg_bl - avg_fd
                    direction = "outperforms" if diff > 0 else "underperforms"
                    report.append(
                        f"  -> Foundation {direction} baseline by "
                        f"{abs(diff) * 100:.1f}%"
                    )

    # ── Few-shot ─────────────────────────────────────────────────────
    if os.path.exists("results/few_shot_results.csv"):
        df_fs = pd.read_csv("results/few_shot_results.csv")
        report.append("\n" + "=" * 70)
        report.append("FEW-SHOT EVALUATION")
        report.append("=" * 70)
        for frac in sorted(df_fs["fraction"].unique()):
            sub = df_fs[df_fs["fraction"] == frac]
            for model_name in ["baseline", "foundation"]:
                model_sub = sub[sub["model"] == model_name]
                if not model_sub.empty:
                    acc = model_sub["accuracy"].mean()
                    report.append(
                        f"  {int(frac * 100):3d}% data | "
                        f"{model_name:12s}: avg_acc={acc:.4f}"
                    )

    # ── Leave-one-out ────────────────────────────────────────────────
    if os.path.exists("results/leave_one_out_results.csv"):
        df_loo = pd.read_csv("results/leave_one_out_results.csv")
        report.append("\n" + "=" * 70)
        report.append("CROSS-DOMAIN GENERALIZATION (Leave-One-Out)")
        report.append("=" * 70)
        report.append(df_loo.to_string(index=False))

    # ── Ablations ────────────────────────────────────────────────────
    if os.path.exists("results/ablation_results.csv"):
        df_abl = pd.read_csv("results/ablation_results.csv")
        report.append("\n" + "=" * 70)
        report.append("ABLATION STUDIES")
        report.append("=" * 70)
        for abl in df_abl["ablation"].unique():
            sub = df_abl[df_abl["ablation"] == abl]
            if "accuracy" in sub.columns:
                valid = sub[sub["accuracy"] > 0]["accuracy"]
                if len(valid) > 0:
                    report.append(
                        f"  {abl:30s}: avg_acc={valid.mean():.4f}"
                    )
            if "rul_mae" in sub.columns:
                valid = sub[sub["rul_mae"] > 0]["rul_mae"]
                if len(valid) > 0:
                    report.append(
                        f"  {abl:30s}: avg_mae={valid.mean():.4f}"
                    )

    # ── Pre-training log ─────────────────────────────────────────────
    if os.path.exists("results/pretrain_log.csv"):
        df_pt = pd.read_csv("results/pretrain_log.csv")
        report.append("\n" + "=" * 70)
        report.append("PRE-TRAINING SUMMARY")
        report.append("=" * 70)
        report.append(
            f"  Epochs trained: {len(df_pt)}"
        )
        report.append(
            f"  Final train loss: {df_pt['train_loss'].iloc[-1]:.6f}"
        )
        report.append(
            f"  Best val loss:    {df_pt['val_loss'].min():.6f}"
        )

    report_text = "\n".join(report)
    print(report_text)

    with open("results/summary_report.txt", "w") as f:
        f.write(report_text)
    print(f"\nFull report saved -> results/summary_report.txt")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PHM Foundation Model — End-to-End Pipeline",
    )
    parser.add_argument(
        "--skip-ablations", action="store_true",
        help="Skip ablation studies for faster execution",
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="Run only a specific step (1-7). 0 = run all.",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: skip few-shot and ablations",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(
        "results", "plots", "ablation_plots", "checkpoints", "data",
    )

    t0 = time.time()

    steps = {
        1: ("Data Pipeline", step1_data),
        2: ("Baseline Training", step2_baseline),
        3: ("Self-Supervised Pre-Training", step3_pretrain),
        4: ("Fine-Tuning", step4_finetune),
        5: ("Evaluation", step5_evaluation),
        6: ("Ablation Studies", step6_ablations),
        7: ("Summary Report", step7_summary),
    }

    if args.step > 0:
        if args.step in steps:
            name, fn = steps[args.step]
            print(f"\nRunning step {args.step}: {name}")
            fn()
        else:
            print(f"Invalid step: {args.step}. Choose 1-7.")
            sys.exit(1)
    else:
        step1_data()
        step2_baseline()
        step3_pretrain()
        step4_finetune()

        if args.quick:
            print("\n[Quick mode] Skipping few-shot and ablations.")
        else:
            step5_evaluation()
            if not args.skip_ablations:
                step6_ablations()

        step7_summary()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE — Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")
    print(f"\nOutputs:")
    print(f"  results/pretrain_log.csv          Pre-training loss curve")
    print(f"  results/baseline_metrics.csv      Baseline CNN results")
    print(f"  results/foundation_metrics.csv    Foundation model results")
    print(f"  results/comparison_table.csv      Side-by-side comparison")
    print(f"  results/few_shot_results.csv      Few-shot evaluation")
    print(f"  results/leave_one_out_results.csv Cross-domain transfer")
    print(f"  results/ablation_results.csv      Ablation studies")
    print(f"  results/summary_report.txt        Summary report")
    print(f"  plots/accuracy_comparison.png     Accuracy bar chart")
    print(f"  plots/f1_comparison.png           F1 bar chart")
    print(f"  plots/few_shot_comparison.png     Few-shot curves")
    print(f"  plots/tsne_by_domain.png          t-SNE by domain")
    print(f"  plots/tsne_by_fault.png           t-SNE by fault type")
    print(f"  ablation_plots/                   Ablation visualizations")
    print(f"  checkpoints/pretrained_encoder.pt Pre-trained weights")


if __name__ == "__main__":
    main()
