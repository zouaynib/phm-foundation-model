"""
Ablation Studies — Systematic Component Analysis
=================================================

Quantifies the contribution of each architectural component:

  1. **No FreqCondNorm** — standard LayerNorm (no frequency conditioning).
  2. **No dataset embedding** — remove dataset ID embedding.
  3. **No pre-training** — random initialisation (no masked autoencoding).
  4. **Mask ratio sweep** — 15%, 30%, 40%, 50%, 60%, 75%.
  5. **Patch size sweep** — 32, 64, 128.
  6. **Encoder depth sweep** — 2, 4, 6 Transformer layers.

Each ablation trains a variant and evaluates on all datasets.
Results are saved to results/ablation_results.csv with plots.

Usage:
    python ablation_studies.py
    python ablation_studies.py --config configs/config.yaml
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_config, set_seed, get_device, ensure_dirs,
    EarlyStopping, Timer,
)
from data_pipeline import (
    PHMDataset, make_loader, get_split_indices, get_all_split_indices,
)
from foundation_model import FoundationModel
import h5py


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _get_task_info(ds_cfg):
    has_cls, has_rul, num_classes = False, False, 0
    for task in ds_cfg.get("tasks", []):
        if task["type"] == "classification":
            has_cls = True
            num_classes = task["num_classes"]
        elif task["type"] == "regression":
            has_rul = True
    return has_cls, has_rul, num_classes


def _build_model(cfg, device, use_freq_cond=True, use_dataset_embed=True,
                 override_num_layers=None, override_patch_size=None,
                 override_patch_stride=None):
    """Build foundation model with optional overrides for ablation.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.
    device : torch.device
    use_freq_cond : bool
        Whether to use FreqCondNorm (False -> standard LayerNorm).
    use_dataset_embed : bool
        Whether to use dataset ID embedding.
    override_num_layers, override_patch_size, override_patch_stride : int
        Override specific architecture parameters.

    Returns
    -------
    FoundationModel
    """
    fcfg = cfg["foundation"]
    pcfg = cfg["pretrain"]
    L = cfg["data"]["window_length"]
    max_channels = max(d["num_channels"] for d in cfg["datasets"])

    ds_model_configs = [
        {"num_channels": d["num_channels"], "tasks": d["tasks"]}
        for d in cfg["datasets"]
    ]

    patch_size = override_patch_size or fcfg["patch_size"]
    patch_stride = override_patch_stride or (patch_size // 2)

    return FoundationModel(
        dataset_configs=ds_model_configs,
        window_length=L,
        d_model=fcfg["d_model"],
        patch_size=patch_size,
        patch_stride=patch_stride,
        num_heads=fcfg["num_heads"],
        num_layers=override_num_layers or fcfg["num_layers"],
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
        use_freq_cond=use_freq_cond,
        use_dataset_embed=use_dataset_embed,
    ).to(device)


def _quick_pretrain_mae(model, cfg, device, epochs=15, mask_ratio=0.4):
    """Quick masked autoencoder pre-training for ablation.

    Uses fewer epochs than full pre-training to keep ablation runtime
    reasonable while still capturing the effect of each component.

    Parameters
    ----------
    model : FoundationModel
    cfg : dict
    device : torch.device
    epochs : int
    mask_ratio : float

    Returns
    -------
    FoundationModel
        Pre-trained model.
    """
    hdf5_path = cfg["data"]["combined_hdf5"]
    pcfg = cfg["pretrain"]
    num_ds = len(cfg["datasets"])

    tr_idx, _, _ = get_all_split_indices(
        hdf5_path, num_ds,
        cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"],
    )

    # Balanced sampler
    with h5py.File(hdf5_path, "r") as f:
        dsids = f["dataset_id"][:][tr_idx]
    unique, counts = np.unique(dsids, return_counts=True)
    weight_map = {uid: 1.0 / c for uid, c in zip(unique, counts)}
    weights = np.array([weight_map[d] for d in dsids])
    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
    )

    train_ds = PHMDataset(hdf5_path, indices=tr_idx)
    train_loader = DataLoader(
        train_ds, batch_size=pcfg["batch_size"],
        sampler=sampler, num_workers=0, pin_memory=True,
    )

    pretrain_params = model.get_backbone_params() + model.get_decoder_params()
    optimizer = AdamW(
        pretrain_params, lr=pcfg["lr"], weight_decay=pcfg["weight_decay"],
    )

    model.train()
    for epoch in range(1, epochs + 1):
        for sigs, _lbls, _rul, freqs, _dsids, nchannels in train_loader:
            sigs = sigs.to(device)
            freqs = freqs.to(device, dtype=torch.float32)
            nchannels = nchannels.to(device)

            optimizer.zero_grad()
            loss, _ = model.forward_pretrain(
                sigs, freqs, nchannels, mask_ratio,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def _quick_finetune_and_eval(model, cfg, device, ds_idx, epochs=15):
    """Quick per-dataset fine-tune and evaluate for ablation.

    Parameters
    ----------
    model : FoundationModel
    cfg : dict
    device : torch.device
    ds_idx : int
    epochs : int

    Returns
    -------
    dict
        Test metrics.
    """
    hdf5_path = cfg["data"]["combined_hdf5"]
    ds_cfg = cfg["datasets"][ds_idx]
    has_cls, has_rul, _ = _get_task_info(ds_cfg)
    n_ch = ds_cfg["num_channels"]
    freq = ds_cfg["original_sampling_freq"]
    bs = cfg["baseline"]["batch_size"]

    tr_idx, va_idx, te_idx = get_split_indices(
        hdf5_path, ds_idx,
        cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"],
    )

    train_loader = make_loader(hdf5_path, tr_idx, bs, shuffle=True)
    test_loader = make_loader(hdf5_path, te_idx, bs, shuffle=False)

    # Freeze backbone, train heads
    for p in model.get_backbone_params():
        p.requires_grad = False
    for p in model.get_decoder_params():
        p.requires_grad = False

    head_params = model.get_head_params(ds_idx)
    embed_params = model.get_embed_params()
    trainable = head_params + embed_params
    if not trainable:
        return {}

    cls_criterion = nn.CrossEntropyLoss()
    rul_criterion = nn.SmoothL1Loss()
    opt = AdamW(trainable, lr=0.001)
    sched = CosineAnnealingLR(opt, T_max=epochs)

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

    # Evaluate
    model.eval()
    cls_preds, cls_labels = [], []
    rul_preds_list, rul_tgts_list = [], []

    with torch.no_grad():
        for sigs, lbls, rul_targets, *_ in test_loader:
            sigs = sigs.to(device)
            cls_logits, rul_preds = model.forward_single_dataset(
                sigs, freq, ds_idx, n_ch,
            )
            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    cls_preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                    cls_labels.extend(lbls[valid].numpy())
            if has_rul and rul_preds is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    rul_preds_list.extend(rul_preds[valid].cpu().numpy())
                    rul_tgts_list.extend(rul_targets[valid].numpy())

    results = {}
    if cls_preds:
        results["accuracy"] = accuracy_score(cls_labels, cls_preds)
    if rul_preds_list:
        results["rul_mae"] = float(np.mean(
            np.abs(np.array(rul_preds_list) - np.array(rul_tgts_list))))
    return results


# ═════════════════════════════════════════════════════════════════════
# Ablation 1: No FreqCondNorm
# ═════════════════════════════════════════════════════════════════════

def ablation_no_freq_cond(cfg, device):
    """Replace FreqCondNorm with standard LayerNorm."""
    print("\n--- Ablation: No FreqCondNorm (standard LayerNorm) ---")
    model = _build_model(cfg, device, use_freq_cond=False)
    model = _quick_pretrain_mae(model, cfg, device)

    results = []
    for ds_idx in range(len(cfg["datasets"])):
        r = _quick_finetune_and_eval(model, cfg, device, ds_idx)
        r["dataset"] = cfg["datasets"][ds_idx]["name"]
        r["ablation"] = "no_freq_cond"
        results.append(r)
    return results


# ═════════════════════════════════════════════════════════════════════
# Ablation 2: No Dataset Embedding
# ═════════════════════════════════════════════════════════════════════

def ablation_no_dataset_embed(cfg, device):
    """Remove dataset ID embedding."""
    print("\n--- Ablation: No Dataset Embedding ---")
    model = _build_model(cfg, device, use_dataset_embed=False)
    model = _quick_pretrain_mae(model, cfg, device)

    results = []
    for ds_idx in range(len(cfg["datasets"])):
        r = _quick_finetune_and_eval(model, cfg, device, ds_idx)
        r["dataset"] = cfg["datasets"][ds_idx]["name"]
        r["ablation"] = "no_dataset_embed"
        results.append(r)
    return results


# ═════════════════════════════════════════════════════════════════════
# Ablation 3: No Pre-Training (Random Init)
# ═════════════════════════════════════════════════════════════════════

def ablation_no_pretraining(cfg, device):
    """No masked pre-training — fine-tune from random initialisation."""
    print("\n--- Ablation: No Pre-Training (Random Init) ---")
    model = _build_model(cfg, device)
    # Skip pre-training entirely

    results = []
    for ds_idx in range(len(cfg["datasets"])):
        r = _quick_finetune_and_eval(model, cfg, device, ds_idx)
        r["dataset"] = cfg["datasets"][ds_idx]["name"]
        r["ablation"] = "no_pretraining"
        results.append(r)
    return results


# ═════════════════════════════════════════════════════════════════════
# Ablation 4: Mask Ratio Sweep
# ═════════════════════════════════════════════════════════════════════

def ablation_mask_ratios(cfg, device):
    """Sweep mask ratios: how much masking is optimal?"""
    print("\n--- Ablation: Mask Ratio Sweep ---")
    mask_ratios = cfg["ablation"].get(
        "mask_ratios", [0.15, 0.3, 0.4, 0.5, 0.6, 0.75],
    )

    all_results = []
    for mr in mask_ratios:
        print(f"  Mask ratio = {mr}")
        model = _build_model(cfg, device)
        model = _quick_pretrain_mae(model, cfg, device, mask_ratio=mr)

        for ds_idx in range(len(cfg["datasets"])):
            r = _quick_finetune_and_eval(model, cfg, device, ds_idx)
            r["dataset"] = cfg["datasets"][ds_idx]["name"]
            r["ablation"] = f"mask_ratio_{mr}"
            all_results.append(r)

    return all_results


# ═════════════════════════════════════════════════════════════════════
# Ablation 5: Patch Size Sweep
# ═════════════════════════════════════════════════════════════════════

def ablation_patch_sizes(cfg, device):
    """Sweep patch sizes."""
    print("\n--- Ablation: Patch Size Sweep ---")
    patch_sizes = cfg["ablation"].get("patch_sizes", [32, 64, 128])

    all_results = []
    for ps in patch_sizes:
        print(f"  Patch size = {ps}")
        model = _build_model(
            cfg, device,
            override_patch_size=ps, override_patch_stride=ps // 2,
        )
        model = _quick_pretrain_mae(model, cfg, device)

        for ds_idx in range(len(cfg["datasets"])):
            r = _quick_finetune_and_eval(model, cfg, device, ds_idx)
            r["dataset"] = cfg["datasets"][ds_idx]["name"]
            r["ablation"] = f"patch_size_{ps}"
            all_results.append(r)

    return all_results


# ═════════════════════════════════════════════════════════════════════
# Ablation 6: Encoder Depth Sweep
# ═════════════════════════════════════════════════════════════════════

def ablation_num_layers(cfg, device):
    """Sweep number of Transformer layers."""
    print("\n--- Ablation: Encoder Depth Sweep ---")
    layer_counts = cfg["ablation"].get("num_layers_sweep", [2, 4, 6])

    all_results = []
    for nl in layer_counts:
        print(f"  Num layers = {nl}")
        model = _build_model(cfg, device, override_num_layers=nl)
        model = _quick_pretrain_mae(model, cfg, device)

        for ds_idx in range(len(cfg["datasets"])):
            r = _quick_finetune_and_eval(model, cfg, device, ds_idx)
            r["dataset"] = cfg["datasets"][ds_idx]["name"]
            r["ablation"] = f"num_layers_{nl}"
            all_results.append(r)

    return all_results


# ═════════════════════════════════════════════════════════════════════
# Run All Ablations
# ═════════════════════════════════════════════════════════════════════

def run_ablations(config_path="configs/config.yaml"):
    """Run the complete ablation suite.

    Executes all 6 ablation studies and saves results to
    results/ablation_results.csv with summary plots.

    Parameters
    ----------
    config_path : str
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs("results", "ablation_plots")

    all_rows = []

    # 1. No FreqCondNorm
    all_rows.extend(ablation_no_freq_cond(cfg, device))

    # 2. No dataset embedding
    all_rows.extend(ablation_no_dataset_embed(cfg, device))

    # 3. No pre-training
    all_rows.extend(ablation_no_pretraining(cfg, device))

    # 4. Mask ratio sweep
    all_rows.extend(ablation_mask_ratios(cfg, device))

    # 5. Patch size sweep
    all_rows.extend(ablation_patch_sizes(cfg, device))

    # 6. Encoder depth sweep
    all_rows.extend(ablation_num_layers(cfg, device))

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv("results/ablation_results.csv", index=False)
    print(f"\nAblation results -> results/ablation_results.csv")

    # Plots
    _plot_ablation_summary(df)
    return df


def _plot_ablation_summary(df):
    """Generate ablation study visualizations."""
    ensure_dirs("ablation_plots")

    # ── Component ablation bar chart ─────────────────────────────────
    component_names = ["no_freq_cond", "no_dataset_embed", "no_pretraining"]
    comp_df = df[df["ablation"].isin(component_names)]

    if not comp_df.empty and "accuracy" in comp_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        avg_accs = []
        labels = []
        for abl in component_names:
            sub = comp_df[comp_df["ablation"] == abl]
            valid = sub[sub["accuracy"] > 0]["accuracy"]
            if len(valid) > 0:
                avg_accs.append(valid.mean())
                labels.append(abl.replace("_", " ").title())

        if avg_accs:
            y_pos = np.arange(len(labels))
            colors = ["#E74C3C", "#F39C12", "#3498DB"][:len(labels)]
            bars = ax.barh(y_pos, avg_accs, color=colors, alpha=0.8)
            for bar, val in zip(bars, avg_accs):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=10)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Average Accuracy (Classification Datasets)")
            ax.set_title("Component Ablation: Effect of Removing Each Component")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig("ablation_plots/component_ablation.png", dpi=150)
            plt.close()

    # ── Mask ratio sweep ─────────────────────────────────────────────
    mr_df = df[df["ablation"].str.startswith("mask_ratio_")]
    if not mr_df.empty and "accuracy" in mr_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = mr_df.groupby("ablation")["accuracy"].mean()
        ratios = [float(k.split("_")[-1]) for k in groups.index]
        sorted_pairs = sorted(zip(ratios, groups.values))
        x_vals = [p[0] for p in sorted_pairs]
        y_vals = [p[1] for p in sorted_pairs]
        ax.plot(x_vals, y_vals, "o-", color="#E74C3C", linewidth=2,
                markersize=8)
        ax.set_xlabel("Mask Ratio")
        ax.set_ylabel("Average Accuracy")
        ax.set_title("Effect of Mask Ratio on Downstream Performance")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("ablation_plots/mask_ratio_sweep.png", dpi=150)
        plt.close()

    # ── Patch size sweep ─────────────────────────────────────────────
    ps_df = df[df["ablation"].str.startswith("patch_size_")]
    if not ps_df.empty and "accuracy" in ps_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = ps_df.groupby("ablation")["accuracy"].mean()
        sizes = [int(k.split("_")[-1]) for k in groups.index]
        sorted_pairs = sorted(zip(sizes, groups.values))
        ax.plot([p[0] for p in sorted_pairs],
                [p[1] for p in sorted_pairs],
                "s-", color="#DD8452", linewidth=2, markersize=8)
        ax.set_xlabel("Patch Size")
        ax.set_ylabel("Average Accuracy")
        ax.set_title("Patch Size Sweep")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("ablation_plots/patch_size_sweep.png", dpi=150)
        plt.close()

    # ── Encoder depth sweep ──────────────────────────────────────────
    nl_df = df[df["ablation"].str.startswith("num_layers_")]
    if not nl_df.empty and "accuracy" in nl_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = nl_df.groupby("ablation")["accuracy"].mean()
        layers = [int(k.split("_")[-1]) for k in groups.index]
        sorted_pairs = sorted(zip(layers, groups.values))
        ax.plot([p[0] for p in sorted_pairs],
                [p[1] for p in sorted_pairs],
                "D-", color="#55A868", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Transformer Layers")
        ax.set_ylabel("Average Accuracy")
        ax.set_title("Encoder Depth Sweep")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("ablation_plots/encoder_depth_sweep.png", dpi=150)
        plt.close()

    print("  Ablation plots -> ablation_plots/")


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    run_ablations(args.config)
