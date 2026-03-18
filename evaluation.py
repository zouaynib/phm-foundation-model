"""
Evaluation — Comprehensive Benchmarking Suite
==============================================

Produces all result tables and figures for the paper:

  1. **Comparison table**: Baseline CNN vs Foundation Model (full data).
  2. **Few-shot evaluation**: 1%, 5%, 10%, 50%, 100% labeled data with
     multiple random seeds for statistical significance.
  3. **Cross-domain transfer**: Leave-one-domain-out zero-shot evaluation
     with optional linear probing.
  4. **t-SNE visualization**: Learned representations colored by fault
     type across domains.
  5. **Comparison plots**: Bar charts for accuracy, F1, RUL MAE.

Usage:
    python evaluation.py
    python evaluation.py --config configs/config.yaml
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_config, set_seed, get_device, ensure_dirs,
    EarlyStopping, Timer, RUL_SENTINEL, CLS_SENTINEL,
    compute_rul_metrics, load_pretrained_flexible,
)
from data_pipeline import (
    PHMDataset, make_loader, get_split_indices, get_all_split_indices,
)
from baseline_model import BaselineCNN
from foundation_model import FoundationModel
from torch.utils.data import DataLoader
import h5py


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _get_task_info(ds_cfg):
    """Extract task type info from dataset config."""
    has_cls, has_rul, num_classes = False, False, 0
    for task in ds_cfg.get("tasks", []):
        if task["type"] == "classification":
            has_cls = True
            num_classes = task["num_classes"]
        elif task["type"] == "regression":
            has_rul = True
    return has_cls, has_rul, num_classes


def _build_foundation_model(cfg, device):
    """Construct FoundationModel from config."""
    fcfg = cfg["foundation"]
    pcfg = cfg["pretrain"]
    L = cfg["data"]["window_length"]
    max_channels = max(d["num_channels"] for d in cfg["datasets"])

    ds_model_configs = [
        {"num_channels": d["num_channels"], "tasks": d["tasks"]}
        for d in cfg["datasets"]
    ]

    return FoundationModel(
        dataset_configs=ds_model_configs,
        window_length=L,
        d_model=fcfg["d_model"],
        patch_size=fcfg["patch_size"],
        patch_stride=fcfg["patch_stride"],
        num_heads=fcfg["num_heads"],
        num_layers=fcfg["num_layers"],
        dim_feedforward=fcfg["dim_feedforward"],
        dropout=fcfg["dropout"],
        freq_dim=fcfg["freq_dim"],
        dataset_embed_dim=fcfg["dataset_embed_dim"],
        latent_dim=fcfg["latent_dim"],
        max_channels=max_channels,
        decoder_d_model=pcfg["decoder_d_model"],
        decoder_num_heads=pcfg["decoder_num_heads"],
        decoder_num_layers=pcfg["decoder_num_layers"],
        decoder_dim_feedforward=pcfg["decoder_dim_feedforward"],
    ).to(device)


# ═════════════════════════════════════════════════════════════════════
# 1. COMPARISON TABLE (Baseline vs Foundation, full data)
# ═════════════════════════════════════════════════════════════════════

def comparison_table(config_path="configs/config.yaml"):
    """Merge baseline and foundation results into one comparison table.

    Reads CSV files produced by train_baseline.py and fine_tune.py,
    computes accuracy/F1/MAE gains, saves to results/comparison_table.csv.

    Returns
    -------
    pd.DataFrame
        Merged comparison table.
    """
    bl_path = "results/baseline_metrics.csv"
    fd_path = "results/foundation_metrics.csv"

    if not os.path.exists(bl_path):
        print(f"  Warning: {bl_path} not found — skipping comparison.")
        return pd.DataFrame()
    if not os.path.exists(fd_path):
        print(f"  Warning: {fd_path} not found — skipping comparison.")
        return pd.DataFrame()

    bl = pd.read_csv(bl_path)
    fd = pd.read_csv(fd_path)
    merged = bl.merge(fd, on="dataset", suffixes=("_baseline", "_foundation"))

    # Build column list dynamically
    cols = ["dataset"]
    rename = {"dataset": "Dataset"}

    for col_base in ["accuracy", "f1_score", "rul_mae", "rul_rmse"]:
        bl_col = f"{col_base}_baseline"
        fd_col = f"{col_base}_foundation"
        if bl_col in merged.columns and fd_col in merged.columns:
            cols.extend([bl_col, fd_col])
            nice = col_base.replace("_", " ").title()
            rename[bl_col] = f"Baseline {nice}"
            rename[fd_col] = f"Foundation {nice}"

    merged = merged[[c for c in cols if c in merged.columns]]
    merged.rename(columns=rename, inplace=True)

    # Compute gain columns
    if "Baseline Accuracy" in merged.columns:
        merged["Acc Gain"] = (
            merged["Foundation Accuracy"] - merged["Baseline Accuracy"]
        )
    if "Baseline F1 Score" in merged.columns:
        merged["F1 Gain"] = (
            merged["Foundation F1 Score"] - merged["Baseline F1 Score"]
        )
    if "Baseline Rul Mae" in merged.columns:
        merged["MAE Gain"] = (
            merged["Baseline Rul Mae"] - merged["Foundation Rul Mae"]
        )

    merged.to_csv("results/comparison_table.csv", index=False)
    print("\n=== COMPARISON TABLE ===")
    print(merged.to_string(index=False))
    return merged


def plot_comparison(merged_df):
    """Generate bar chart comparisons of baseline vs foundation."""
    if merged_df.empty:
        print("  No comparison data — skipping plots.")
        return
    ensure_dirs("plots")

    w = 0.35

    # Helper for creating a comparison bar chart
    def _bar_chart(col_bl, col_fd, ylabel, title, filename, lower_better=False):
        if col_bl not in merged_df.columns or col_fd not in merged_df.columns:
            return
        mask = merged_df[col_bl] > 0
        if not mask.any():
            return
        sub = merged_df[mask]
        x = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w / 2, sub[col_bl], w, label="Baseline CNN",
               color="#4C72B0")
        ax.bar(x + w / 2, sub[col_fd], w, label="Foundation Model",
               color="#DD8452")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Dataset"].values, rotation=30, ha="right")
        ax.legend()
        if not lower_better:
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/{filename}", dpi=150)
        plt.close()

    _bar_chart("Baseline Accuracy", "Foundation Accuracy",
               "Accuracy", "Test Accuracy: Baseline vs Foundation",
               "accuracy_comparison.png")

    _bar_chart("Baseline F1 Score", "Foundation F1 Score",
               "Macro F1-Score", "Test F1: Baseline vs Foundation",
               "f1_comparison.png")

    _bar_chart("Baseline Rul Mae", "Foundation Rul Mae",
               "RUL MAE (normalized)", "RUL MAE: Baseline vs Foundation",
               "rul_mae_comparison.png", lower_better=True)

    print("  Comparison plots -> plots/")


# ═════════════════════════════════════════════════════════════════════
# 2. FEW-SHOT EVALUATION
# ═════════════════════════════════════════════════════════════════════

def _compute_val_metric(val_results, has_cls, has_rul):
    """Compute a scalar validation metric for early stopping.

    For classification: accuracy (higher = better).
    For regression: 1.0 - MAE (higher = better, avoids negative metrics).
    For both: accuracy + (1.0 - MAE).
    """
    metric = 0.0
    if has_cls and "acc" in val_results:
        metric += val_results["acc"]
    if has_rul and "rul_mae" in val_results:
        # Use (1 - MAE) so metric is always positive and higher = better
        metric += (1.0 - val_results["rul_mae"])
    return metric


def _run_fewshot_stage(model, train_loader, val_loader, device,
                       ds_idx, freq, n_ch, has_cls, has_rul,
                       trainable_params, epochs, lr, patience):
    """Run one few-shot fine-tuning stage. Returns best val metric."""
    from fine_tune import evaluate_single
    cls_criterion = nn.CrossEntropyLoss()
    rul_criterion = nn.SmoothL1Loss()

    opt = AdamW(trainable_params, lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    es = EarlyStopping(patience=patience)
    best_metric, best_state = -float("inf"), None

    for ep in range(1, epochs + 1):
        model.train()
        for sigs, lbls, rul_targets, *_ in train_loader:
            sigs = sigs.to(device)
            lbls = lbls.to(device)
            rul_targets = rul_targets.to(device, dtype=torch.float32)
            opt.zero_grad()
            cls_logits, rul_preds = model.forward_single_dataset(
                sigs, freq, ds_idx, n_ch,
            )
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    loss = loss + cls_criterion(cls_logits[valid], lbls[valid])
            if has_rul and rul_preds is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    loss = loss + rul_criterion(
                        rul_preds[valid], rul_targets[valid],
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        val_results = evaluate_single(
            model, val_loader, device, ds_idx, freq, n_ch, has_cls, has_rul,
        )
        val_metric = _compute_val_metric(val_results, has_cls, has_rul)

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if es.step(val_metric):
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return best_metric


def _quick_finetune_foundation(model, train_loader, val_loader, device,
                               ds_idx, freq, n_ch, has_cls, has_rul,
                               epochs=30, lr=0.0005, patience=8):
    """Two-stage fine-tune for few-shot evaluation.

    Stage 1: Train heads + projector + embeddings (frozen backbone).
    Stage 2: Partially unfreeze last 2 encoder layers and continue.

    This mirrors the full 3-stage fine-tuning approach but is faster.
    """
    # Freeze backbone + decoder
    for p in model.get_backbone_params():
        p.requires_grad = False
    for p in model.get_decoder_params():
        p.requires_grad = False

    # ── Stage 1: Heads + projector + embeddings ──────────────────
    head_params = model.get_head_params(ds_idx)
    embed_params = model.get_embed_params()  # includes projector
    stage1_params = head_params + embed_params
    if not stage1_params:
        return {}

    s1_epochs = max(epochs // 2, 10)
    _run_fewshot_stage(
        model, train_loader, val_loader, device,
        ds_idx, freq, n_ch, has_cls, has_rul,
        stage1_params, s1_epochs, lr, patience,
    )

    # ── Stage 2: Partially unfreeze last 2 encoder layers ────────
    encoder_layers = list(model.encoder.layers)
    for layer in encoder_layers[-2:]:
        for p in layer.parameters():
            p.requires_grad = True

    stage2_params = [p for p in model.parameters() if p.requires_grad]
    s2_epochs = max(epochs // 2, 10)
    _run_fewshot_stage(
        model, train_loader, val_loader, device,
        ds_idx, freq, n_ch, has_cls, has_rul,
        stage2_params, s2_epochs, lr * 0.1, patience,
    )
    return None  # model is updated in-place


def _quick_baseline(hdf5_path, ds_idx, ds_cfg, train_indices, val_indices,
                    test_indices, cfg, device, epochs=20):
    """Train a quick baseline CNN on a data subset.

    Returns
    -------
    dict
        Test metrics: 'acc', 'f1', 'rul_mae'.
    """
    bcfg = cfg["baseline"]
    L = cfg["data"]["window_length"]
    has_cls, has_rul, num_classes = _get_task_info(ds_cfg)
    n_ch = ds_cfg["num_channels"]
    bs = min(bcfg["batch_size"], len(train_indices))
    if bs < 2:
        return {}

    train_loader = make_loader(hdf5_path, train_indices, bs, shuffle=True)
    val_loader = make_loader(hdf5_path, val_indices, bs, shuffle=False)
    test_loader = make_loader(hdf5_path, test_indices, bs, shuffle=False)

    model = BaselineCNN(
        num_classes=num_classes if has_cls else 0,
        window_length=L, in_channels=n_ch,
        channels=tuple(bcfg["channels"]),
        kernel_size=bcfg["kernel_size"], dropout=bcfg["dropout"],
        has_rul_head=has_rul,
    ).to(device)

    cls_criterion = nn.CrossEntropyLoss()
    rul_criterion = nn.MSELoss()
    opt = AdamW(model.parameters(), lr=bcfg["lr"],
                weight_decay=bcfg["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs)
    es = EarlyStopping(patience=5)
    best_metric, best_state = -float("inf"), None

    for ep in range(1, epochs + 1):
        model.train()
        for sigs, lbls, rul_targets, *_ in train_loader:
            sigs = sigs[:, :n_ch, :].to(device)
            lbls = lbls.to(device)
            rul_targets = rul_targets.to(device, dtype=torch.float32)
            opt.zero_grad()
            cls_logits, rul_pred = model(sigs)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    loss = loss + cls_criterion(cls_logits[valid], lbls[valid])
            if has_rul and rul_pred is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    loss = loss + rul_criterion(
                        rul_pred[valid], rul_targets[valid],
                    )
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        val_metric = 0.0
        with torch.no_grad():
            preds, labels = [], []
            rul_preds, rul_tgts = [], []
            for sigs, lbls, rul_targets, *_ in val_loader:
                cls_logits, rul_pred = model(sigs[:, :n_ch, :].to(device))
                if has_cls and cls_logits is not None:
                    valid = lbls >= 0
                    if valid.sum() > 0:
                        preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                        labels.extend(lbls[valid].numpy())
                if has_rul and rul_pred is not None:
                    valid = rul_targets >= 0
                    if valid.sum() > 0:
                        rul_preds.extend(rul_pred[valid].cpu().numpy())
                        rul_tgts.extend(rul_targets[valid].numpy())
            if preds:
                val_metric += accuracy_score(labels, preds)
            if rul_preds:
                val_metric += 1.0 - float(np.mean(
                    np.abs(np.array(rul_preds) - np.array(rul_tgts))))

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if es.step(val_metric):
            break

    if best_state is None:
        return {}
    model.load_state_dict(best_state)
    model.to(device).eval()

    results = {}
    with torch.no_grad():
        preds, labels = [], []
        rul_preds, rul_tgts = [], []
        for sigs, lbls, rul_targets, *_ in test_loader:
            cls_logits, rul_pred = model(sigs[:, :n_ch, :].to(device))
            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                    labels.extend(lbls[valid].numpy())
            if has_rul and rul_pred is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    rul_preds.extend(rul_pred[valid].cpu().numpy())
                    rul_tgts.extend(rul_targets[valid].numpy())

    if preds:
        results["acc"] = accuracy_score(labels, preds)
        results["f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    if rul_preds:
        results["rul_mae"] = float(np.mean(
            np.abs(np.array(rul_preds) - np.array(rul_tgts))))
    return results


def few_shot_experiment(config_path="configs/config.yaml"):
    """Few-shot evaluation: train with 1%, 5%, 10%, 50%, 100% labeled data.

    For each fraction and each dataset:
      - Trains a baseline CNN from scratch.
      - Fine-tunes the pre-trained foundation model (linear probe).
      - Repeats with multiple seeds for confidence intervals.

    This is the **hero result** of the paper — pre-trained models should
    dramatically outperform from-scratch training at low data fractions.

    Returns
    -------
    pd.DataFrame
        Results with columns: dataset, fraction, seed, model,
        accuracy, f1, rul_mae.
    """
    cfg = load_config(config_path)
    device = get_device()
    hdf5_path = cfg["data"]["combined_hdf5"]
    fscfg = cfg.get("few_shot", {})
    fractions = fscfg.get("fractions", [0.01, 0.05, 0.1, 0.5, 1.0])
    num_seeds = fscfg.get("num_seeds", 3)
    ft_epochs = fscfg.get("finetune_epochs", 30)
    ft_lr = fscfg.get("lr", 0.0005)
    ft_patience = fscfg.get("patience", 8)

    ckpt_path = "checkpoints/pretrained_encoder.pt"
    has_pretrained = os.path.exists(ckpt_path)
    if has_pretrained:
        pretrained_state = torch.load(
            ckpt_path, map_location="cpu", weights_only=True,
        )
    else:
        print("  Warning: No pre-trained model found. "
              "Skipping foundation few-shot.")
        pretrained_state = None

    results = []

    for ds_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        has_cls, has_rul, num_classes = _get_task_info(ds_cfg)
        n_ch = ds_cfg["num_channels"]
        freq = ds_cfg["original_sampling_freq"]
        bs = cfg["baseline"]["batch_size"]

        print(f"\n  Few-shot: {name}")

        for frac in fractions:
            for seed in range(num_seeds):
                set_seed(cfg["seed"] + seed)

                tr_idx, va_idx, te_idx = get_split_indices(
                    hdf5_path, ds_idx,
                    cfg["data"]["train_ratio"], cfg["data"]["val_ratio"],
                    cfg["seed"] + seed,
                )

                # Subsample training data
                n_use = max(2, int(len(tr_idx) * frac))
                rng = np.random.RandomState(cfg["seed"] + seed)
                sub_tr = rng.choice(tr_idx, size=n_use, replace=False)

                # ── Baseline CNN ─────────────────────────────────────
                bl_results = _quick_baseline(
                    hdf5_path, ds_idx, ds_cfg, sub_tr, va_idx, te_idx,
                    cfg, device,
                )

                results.append({
                    "dataset": name, "fraction": frac, "seed": seed,
                    "model": "baseline",
                    "accuracy": round(bl_results.get("acc", 0), 4),
                    "f1": round(bl_results.get("f1", 0), 4),
                    "rul_mae": round(bl_results.get("rul_mae", 0), 4),
                })

                # ── Foundation (pre-trained + fine-tuned) ────────────
                if pretrained_state is not None:
                    model = _build_foundation_model(cfg, device)
                    load_pretrained_flexible(model, pretrained_state)

                    train_loader = make_loader(
                        hdf5_path, sub_tr, bs, shuffle=True,
                    )
                    val_loader = make_loader(
                        hdf5_path, va_idx, bs, shuffle=False,
                    )
                    test_loader = make_loader(
                        hdf5_path, te_idx, bs, shuffle=False,
                    )

                    _quick_finetune_foundation(
                        model, train_loader, val_loader, device,
                        ds_idx, freq, n_ch, has_cls, has_rul,
                        epochs=ft_epochs, lr=ft_lr, patience=ft_patience,
                    )

                    from fine_tune import evaluate_single
                    fd_results = evaluate_single(
                        model, test_loader, device, ds_idx, freq, n_ch,
                        has_cls, has_rul,
                    )

                    results.append({
                        "dataset": name, "fraction": frac, "seed": seed,
                        "model": "foundation",
                        "accuracy": round(fd_results.get("acc", 0), 4),
                        "f1": round(fd_results.get("f1", 0), 4),
                        "rul_mae": round(fd_results.get("rul_mae", 0), 4),
                    })

                print(f"    frac={frac:.2f} seed={seed}: "
                      f"BL_acc={bl_results.get('acc', 0):.3f}  "
                      f"FD_acc={fd_results.get('acc', 0):.3f}" if pretrained_state else "")

    # Save and plot
    df = pd.DataFrame(results)
    df.to_csv("results/few_shot_results.csv", index=False)
    print("\n  Few-shot results -> results/few_shot_results.csv")

    _plot_few_shot(df, cfg)
    return df


def _plot_few_shot(df, cfg):
    """Plot few-shot results: accuracy vs data fraction per dataset."""
    ensure_dirs("plots")

    datasets = cfg["datasets"]
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for i, ds_cfg in enumerate(datasets):
        name = ds_cfg["name"]
        has_cls, has_rul, _ = _get_task_info(ds_cfg)
        metric_col = "accuracy" if has_cls else "rul_mae"
        ylabel = "Accuracy" if has_cls else "1 - MAE"

        sub = df[df["dataset"] == name]

        for model_name, color, marker in [
            ("baseline", "#4C72B0", "o"),
            ("foundation", "#DD8452", "s"),
        ]:
            model_sub = sub[sub["model"] == model_name]
            if model_sub.empty:
                continue

            # Aggregate across seeds: mean ± std
            agg = model_sub.groupby("fraction")[metric_col].agg(
                ["mean", "std"],
            )
            x = agg.index.values
            y = agg["mean"].values
            if not has_cls:
                y = 1 - y  # Plot 1-MAE for regression
            yerr = agg["std"].values

            axes[i].errorbar(
                x, y, yerr=yerr, fmt=f"{marker}-", color=color,
                label=model_name.title(), capsize=3, markersize=6,
            )

        axes[i].set_title(name, fontsize=11)
        axes[i].set_xlabel("Labeled Data Fraction")
        axes[i].set_xscale("log")
        if i == 0:
            axes[i].set_ylabel(ylabel)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.suptitle("Few-Shot Performance: Baseline vs Foundation", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/few_shot_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Few-shot plot -> plots/few_shot_comparison.png")


# ═════════════════════════════════════════════════════════════════════
# 3. CROSS-DOMAIN TRANSFER (Leave-One-Domain-Out)
# ═════════════════════════════════════════════════════════════════════

def leave_one_out_experiment(config_path="configs/config.yaml"):
    """Leave-one-domain-out: evaluate pre-trained model on held-out domain.

    For each dataset with a classification task:
      - Load the pre-trained encoder (trained on ALL domains jointly).
      - Evaluate on the held-out dataset's test set WITHOUT fine-tuning.
      - Also evaluate with a lightweight linear probe (frozen backbone
        + train a linear head on 10% of the held-out domain's data).

    This tests whether the pre-trained representations transfer across
    industrial domains.

    Returns
    -------
    pd.DataFrame
        Columns: held_out_dataset, zero_shot_acc, zero_shot_f1,
        linear_probe_acc, linear_probe_f1.
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    hdf5_path = cfg["data"]["combined_hdf5"]

    ckpt_path = "checkpoints/pretrained_encoder.pt"
    if not os.path.exists(ckpt_path):
        print("  No pre-trained model found. Skipping leave-one-out.")
        return pd.DataFrame()

    pretrained_state = torch.load(
        ckpt_path, map_location="cpu", weights_only=True,
    )

    results = []

    for held_out_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        has_cls, has_rul, num_classes = _get_task_info(ds_cfg)
        n_ch = ds_cfg["num_channels"]
        freq = ds_cfg["original_sampling_freq"]

        if not has_cls:
            print(f"  Skipping {name} (no classification task)")
            continue

        # Load model with pre-trained weights
        model = _build_foundation_model(cfg, device)
        load_pretrained_flexible(model, pretrained_state)

        tr_idx, va_idx, te_idx = get_split_indices(
            hdf5_path, held_out_idx,
            cfg["data"]["train_ratio"], cfg["data"]["val_ratio"],
            cfg["seed"],
        )
        bs = cfg["baseline"]["batch_size"]
        test_loader = make_loader(hdf5_path, te_idx, bs, shuffle=False)

        # ── Zero-shot (no fine-tuning) ───────────────────────────────
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for sigs, lbls, *_ in test_loader:
                sigs = sigs.to(device)
                cls_logits, _ = model.forward_single_dataset(
                    sigs, freq, held_out_idx, n_ch,
                )
                if cls_logits is not None:
                    valid = lbls >= 0
                    if valid.sum() > 0:
                        preds.extend(
                            cls_logits[valid].argmax(1).cpu().numpy())
                        labels.extend(lbls[valid].numpy())

        zs_acc = accuracy_score(labels, preds) if preds else 0.0
        zs_f1 = f1_score(labels, preds, average="macro",
                         zero_division=0) if preds else 0.0

        # ── Linear probe (10% of held-out data) ─────────────────────
        n_probe = max(2, int(len(tr_idx) * 0.1))
        probe_tr = tr_idx[:n_probe]
        probe_loader = make_loader(hdf5_path, probe_tr, bs, shuffle=True)
        val_loader = make_loader(hdf5_path, va_idx, bs, shuffle=False)

        # Reload fresh pre-trained weights
        load_pretrained_flexible(model, pretrained_state)
        for p in model.get_backbone_params():
            p.requires_grad = False
        for p in model.get_decoder_params():
            p.requires_grad = False

        head_params = model.get_head_params(held_out_idx)
        embed_params = model.get_embed_params()
        if head_params or embed_params:
            opt = AdamW(head_params + embed_params, lr=0.001)
            sched = CosineAnnealingLR(opt, T_max=20)
            cls_criterion = nn.CrossEntropyLoss()

            for ep in range(1, 21):
                model.train()
                for sigs, lbls, *_ in probe_loader:
                    sigs = sigs.to(device)
                    lbls = lbls.to(device)
                    opt.zero_grad()
                    cls_logits, _ = model.forward_single_dataset(
                        sigs, freq, held_out_idx, n_ch,
                    )
                    if cls_logits is not None:
                        valid = lbls >= 0
                        if valid.sum() > 0:
                            loss = cls_criterion(
                                cls_logits[valid], lbls[valid],
                            )
                            loss.backward()
                            opt.step()
                sched.step()

        # Evaluate linear probe
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for sigs, lbls, *_ in test_loader:
                sigs = sigs.to(device)
                cls_logits, _ = model.forward_single_dataset(
                    sigs, freq, held_out_idx, n_ch,
                )
                if cls_logits is not None:
                    valid = lbls >= 0
                    if valid.sum() > 0:
                        preds.extend(
                            cls_logits[valid].argmax(1).cpu().numpy())
                        labels.extend(lbls[valid].numpy())

        lp_acc = accuracy_score(labels, preds) if preds else 0.0
        lp_f1 = f1_score(labels, preds, average="macro",
                         zero_division=0) if preds else 0.0

        results.append({
            "held_out_dataset": name,
            "zero_shot_acc": round(zs_acc, 4),
            "zero_shot_f1": round(zs_f1, 4),
            "linear_probe_acc": round(lp_acc, 4),
            "linear_probe_f1": round(lp_f1, 4),
        })
        print(f"  Leave-out {name}: zero_shot_acc={zs_acc:.4f}, "
              f"linear_probe_acc={lp_acc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results/leave_one_out_results.csv", index=False)
    print("  Leave-one-out results -> results/leave_one_out_results.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# 4. t-SNE VISUALIZATION
# ═════════════════════════════════════════════════════════════════════

def tsne_visualization(config_path="configs/config.yaml"):
    """Generate t-SNE plot of learned representations.

    Extracts backbone features from the pre-trained encoder for all
    test samples, then visualizes with t-SNE.  Points are colored by:
      (a) Dataset domain — shows domain clustering/separation.
      (b) Fault type — shows whether fault signatures cluster across domains.

    Saves two plots to plots/.
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    hdf5_path = cfg["data"]["combined_hdf5"]

    ckpt_path = "checkpoints/pretrained_encoder.pt"
    if not os.path.exists(ckpt_path):
        print("  No pre-trained model found. Skipping t-SNE.")
        return

    model = _build_foundation_model(cfg, device)
    pretrained_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Filter out size-mismatched keys (checkpoint may have been saved with
    # a different config, e.g., different num_classes for a dataset).
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            skipped.append(k)
    if skipped:
        print(f"  t-SNE: skipped {len(skipped)} mismatched keys: {skipped}")
    model.load_state_dict(filtered, strict=False)
    model.eval()

    # Collect representations from test sets
    all_reps = []
    all_ds_ids = []
    all_labels = []
    all_ds_names = []

    max_per_ds = 500  # Cap per dataset for t-SNE speed

    for ds_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        n_ch = ds_cfg["num_channels"]
        freq = ds_cfg["original_sampling_freq"]

        _, _, te_idx = get_split_indices(
            hdf5_path, ds_idx,
            cfg["data"]["train_ratio"], cfg["data"]["val_ratio"],
            cfg["seed"],
        )

        # Limit samples for t-SNE performance
        if len(te_idx) > max_per_ds:
            te_idx = te_idx[:max_per_ds]

        loader = make_loader(
            hdf5_path, te_idx, cfg["baseline"]["batch_size"], shuffle=False,
        )

        with torch.no_grad():
            for sigs, lbls, *_ in loader:
                sigs = sigs.to(device)
                freq_t = torch.full(
                    (sigs.shape[0],), freq, device=device,
                )
                nch_t = torch.full(
                    (sigs.shape[0],), n_ch, dtype=torch.long, device=device,
                )
                reps = model.extract_representations(sigs, freq_t, nch_t)
                all_reps.append(reps.cpu().numpy())
                all_ds_ids.extend([ds_idx] * sigs.shape[0])
                all_labels.extend(lbls.numpy())
                all_ds_names.extend([name] * sigs.shape[0])

    if not all_reps:
        print("  No representations extracted. Skipping t-SNE.")
        return

    reps = np.concatenate(all_reps, axis=0)
    ds_ids = np.array(all_ds_ids)
    labels = np.array(all_labels)
    ds_names = np.array(all_ds_names)

    print(f"  t-SNE: {reps.shape[0]} samples, {reps.shape[1]} dims")

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg["seed"],
                n_iter=1000)
    embedded = tsne.fit_transform(reps)

    ensure_dirs("plots")

    # ── Plot 1: Colored by domain ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_names = list(dict.fromkeys(ds_names))  # preserve order
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_names)))

    for i, name in enumerate(unique_names):
        mask = ds_names == name
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=[colors[i]], label=name, alpha=0.6, s=15,
        )

    ax.legend(fontsize=9, markerscale=2)
    ax.set_title("t-SNE of Pre-Trained Representations (by Domain)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("plots/tsne_by_domain.png", dpi=150)
    plt.close()

    # ── Plot 2: Colored by fault type (classification datasets) ──────
    cls_mask = labels >= 0
    if cls_mask.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = np.unique(labels[cls_mask])
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=[colors[i]], label=f"Class {lbl}", alpha=0.6, s=15,
            )

        ax.legend(fontsize=9, markerscale=2)
        ax.set_title("t-SNE of Pre-Trained Representations (by Fault Type)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig("plots/tsne_by_fault.png", dpi=150)
        plt.close()

    print("  t-SNE plots -> plots/tsne_by_domain.png, tsne_by_fault.png")


# ═════════════════════════════════════════════════════════════════════
# 5. RUN ALL EVALUATIONS
# ═════════════════════════════════════════════════════════════════════

def run_evaluation(config_path="configs/config.yaml"):
    """Run the complete evaluation suite.

    1. Comparison table (baseline vs foundation, full data).
    2. Few-shot evaluation (1%–100% data, multiple seeds).
    3. Leave-one-domain-out (zero-shot + linear probe).
    4. t-SNE visualization.
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUITE")
    print("=" * 60)

    # 1. Comparison table
    print("\n--- Comparison Table ---")
    try:
        merged = comparison_table(config_path)
        plot_comparison(merged)
    except Exception as e:
        print(f"  Warning: Comparison table failed: {e}")

    # 2. Few-shot evaluation
    print("\n--- Few-Shot Evaluation ---")
    try:
        few_shot_experiment(config_path)
    except Exception as e:
        print(f"  Warning: Few-shot evaluation failed: {e}")

    # 3. Leave-one-domain-out
    print("\n--- Leave-One-Domain-Out ---")
    try:
        leave_one_out_experiment(config_path)
    except Exception as e:
        print(f"  Warning: Leave-one-out failed: {e}")

    # 4. t-SNE visualization
    print("\n--- t-SNE Visualization ---")
    try:
        tsne_visualization(config_path)
    except Exception as e:
        print(f"  Warning: t-SNE visualization failed: {e}")

    print("\nEvaluations complete!")


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation suite",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    run_evaluation(args.config)
