"""
Fine-Tune Foundation Model — Three-Stage Transfer Learning
==========================================================

Loads the self-supervised pre-trained encoder and fine-tunes it for
each downstream dataset with a three-stage strategy:

  Stage 1 — **Frozen backbone**: freeze the entire Transformer encoder,
      train only the per-dataset task heads.  This prevents catastrophic
      forgetting of pre-trained representations.

  Stage 2 — **Partial unfreeze**: unfreeze the last 2 encoder layers,
      the dataset embedding, and the projector.  Train with a low LR.

  Stage 3 — **Full fine-tune**: unfreeze all parameters and train with
      a very low LR for final adaptation.

Supports both classification (CrossEntropy) and RUL regression
(SmoothL1Loss) tasks.

Usage:
    python fine_tune.py
    python fine_tune.py --config configs/config.yaml
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

from utils import (
    load_config, set_seed, get_device, ensure_dirs,
    CSVLogger, EarlyStopping, Timer, RUL_SENTINEL, CLS_SENTINEL,
    load_pretrained_flexible,
)
from data_pipeline import make_loader, get_split_indices
from foundation_model import FoundationModel


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _get_task_info(ds_cfg):
    """Extract task type information from dataset config."""
    has_cls, has_rul, num_classes = False, False, 0
    for task in ds_cfg.get("tasks", []):
        if task["type"] == "classification":
            has_cls = True
            num_classes = task["num_classes"]
        elif task["type"] == "regression":
            has_rul = True
    return has_cls, has_rul, num_classes


@torch.no_grad()
def evaluate_single(model, loader, device, ds_idx, freq, n_ch,
                    has_cls, has_rul):
    """Evaluate foundation model on a single dataset's test/val set.

    Parameters
    ----------
    model : FoundationModel
    loader : DataLoader
    device : torch.device
    ds_idx : int
        Dataset index.
    freq : float
        Sampling frequency.
    n_ch : int
        Number of channels.
    has_cls, has_rul : bool
        Whether dataset has classification / RUL tasks.

    Returns
    -------
    dict
        Metrics: 'acc', 'f1' for classification; 'rul_mae', 'rul_rmse'
        for regression.
    """
    model.eval()
    all_cls_preds, all_cls_labels = [], []
    all_rul_preds, all_rul_targets = [], []

    for sigs, lbls, rul_targets, _freqs, _dsids, _nch in loader:
        sigs = sigs.to(device)
        cls_logits, rul_preds = model.forward_single_dataset(
            sigs, freq, ds_idx, n_ch,
        )

        if has_cls and cls_logits is not None:
            valid = lbls >= 0
            if valid.sum() > 0:
                all_cls_preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
                all_cls_labels.extend(lbls[valid].numpy())

        if has_rul and rul_preds is not None:
            valid = rul_targets >= 0
            if valid.sum() > 0:
                all_rul_preds.extend(rul_preds[valid].cpu().numpy())
                all_rul_targets.extend(rul_targets[valid].numpy())

    results = {}
    if all_cls_preds:
        preds = np.array(all_cls_preds)
        labels = np.array(all_cls_labels)
        results["acc"] = accuracy_score(labels, preds)
        results["f1"] = f1_score(labels, preds, average="macro",
                                 zero_division=0)
    if all_rul_preds:
        preds = np.array(all_rul_preds)
        targets = np.array(all_rul_targets)
        results["rul_mae"] = float(np.mean(np.abs(preds - targets)))
        results["rul_rmse"] = float(np.sqrt(np.mean((preds - targets) ** 2)))

    return results


# ─────────────────────────────────────────────────────────────────────
# Single Fine-Tuning Stage
# ─────────────────────────────────────────────────────────────────────

def finetune_stage(model, train_loader, val_loader,
                   cls_criterion, rul_criterion,
                   optimizer, scheduler, device,
                   ds_idx, freq, n_ch, has_cls, has_rul,
                   epochs, patience, stage_name):
    """Run one fine-tuning stage with early stopping.

    Parameters
    ----------
    model : FoundationModel
    train_loader, val_loader : DataLoader
    cls_criterion, rul_criterion : nn.Module
    optimizer : Optimizer
    scheduler : LR scheduler
    device : torch.device
    ds_idx : int
    freq : float
    n_ch : int
    has_cls, has_rul : bool
    epochs : int
    patience : int
    stage_name : str

    Returns
    -------
    best_metric : float
    converge_ep : int
    """
    early_stop = EarlyStopping(patience=patience)
    best_metric = -float("inf")
    best_state = None
    converge_ep = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for sigs, lbls, rul_targets, _freqs, _dsids, _nch in train_loader:
            sigs = sigs.to(device)
            lbls = lbls.to(device)
            rul_targets = rul_targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            cls_logits, rul_preds = model.forward_single_dataset(
                sigs, freq, ds_idx, n_ch,
            )

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            if has_cls and cls_logits is not None:
                valid = lbls >= 0
                if valid.sum() > 0:
                    loss = loss + cls_criterion(
                        cls_logits[valid], lbls[valid],
                    )

            if has_rul and rul_preds is not None:
                valid = rul_targets >= 0
                if valid.sum() > 0:
                    loss = loss + rul_criterion(
                        rul_preds[valid], rul_targets[valid],
                    )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        val_results = evaluate_single(
            model, val_loader, device, ds_idx, freq, n_ch, has_cls, has_rul,
        )

        # Combined metric for early stopping (always positive, higher = better)
        val_metric = 0.0
        if has_cls and "acc" in val_results:
            val_metric += val_results["acc"]
        if has_rul and "rul_mae" in val_results:
            val_metric += (1.0 - val_results["rul_mae"])

        if val_metric > best_metric:
            best_metric = val_metric
            converge_ep = epoch
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if early_stop.step(val_metric):
            break

    # Restore best state
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    print(f"    {stage_name}: best_val_metric={best_metric:.4f} "
          f"@ ep {converge_ep}")
    return best_metric, converge_ep


# ─────────────────────────────────────────────────────────────────────
# Build Model from Config
# ─────────────────────────────────────────────────────────────────────

def _build_model(cfg, device):
    """Build a FoundationModel from the YAML config.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.
    device : torch.device

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

    model = FoundationModel(
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

    return model


# ─────────────────────────────────────────────────────────────────────
# Main Fine-Tuning Function
# ─────────────────────────────────────────────────────────────────────

def fine_tune(config_path="configs/config.yaml"):
    """Three-stage fine-tuning of pre-trained foundation model.

    For each dataset:
      1. Load pre-trained encoder weights.
      2. Stage 1: Freeze backbone, train heads.
      3. Stage 2: Partial unfreeze, low LR.
      4. Stage 3: Full fine-tune, very low LR.
      5. Evaluate on test set and log metrics.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs("results", "checkpoints")

    hdf5_path = cfg["data"]["combined_hdf5"]
    ftcfg = cfg["finetune"]
    bs = cfg["baseline"]["batch_size"]

    # Load pre-trained encoder
    pretrained_state = torch.load(
        "checkpoints/pretrained_encoder.pt",
        map_location="cpu", weights_only=True,
    )

    logger = CSVLogger(
        "results/foundation_metrics.csv",
        ["dataset", "accuracy", "f1_score", "rul_mae", "rul_rmse",
         "train_time_s", "converge_epoch", "ft_stage"],
    )

    print(f"Device: {device}")
    print(f"Pre-trained encoder loaded from checkpoints/pretrained_encoder.pt\n")

    for ds_idx, ds_cfg in enumerate(cfg["datasets"]):
        name = ds_cfg["name"]
        has_cls, has_rul, num_classes = _get_task_info(ds_cfg)
        n_ch = ds_cfg["num_channels"]
        freq = ds_cfg["original_sampling_freq"]

        task_str = []
        if has_cls:
            task_str.append(f"cls({num_classes})")
        if has_rul:
            task_str.append("rul")

        print(f"{'═' * 60}")
        print(f"Fine-tuning [{ds_idx}] {name} — {', '.join(task_str)}")
        print(f"{'═' * 60}")

        # Split data
        tr_idx, va_idx, te_idx = get_split_indices(
            hdf5_path, ds_idx,
            cfg["data"]["train_ratio"], cfg["data"]["val_ratio"],
            cfg["seed"],
        )

        train_loader = make_loader(hdf5_path, tr_idx, bs, shuffle=True)
        val_loader = make_loader(hdf5_path, va_idx, bs, shuffle=False)
        test_loader = make_loader(hdf5_path, te_idx, bs, shuffle=False)

        cls_criterion = nn.CrossEntropyLoss()
        rul_criterion = nn.SmoothL1Loss()

        with Timer() as timer:
            # ── Stage 1: Frozen backbone, train heads only ───────────
            model = _build_model(cfg, device)
            load_pretrained_flexible(model, pretrained_state)

            # Freeze encoder backbone and decoder (not needed for fine-tuning)
            for p in model.get_backbone_params():
                p.requires_grad = False
            for p in model.get_decoder_params():
                p.requires_grad = False

            # KEEP projector and embeddings TRAINABLE in Stage 1.
            # The projector was not trained during pre-training (only
            # encoder+decoder were optimized), so it must be allowed to
            # adapt here.  Freezing it would force heads to learn on
            # random projections — the #1 cause of poor fine-tune results.
            for p in model.projector.parameters():
                p.requires_grad = True
            for p in model.get_embed_params():
                p.requires_grad = True

            # Re-initialize RUL heads (fresh weights for regression)
            if has_rul:
                rul_key = f"rul_{ds_idx}"
                if rul_key in model.rul_heads:
                    for m in model.rul_heads[rul_key].modules():
                        if isinstance(m, nn.Linear):
                            nn.init.kaiming_normal_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)

            head_params = model.get_head_params(ds_idx)
            embed_params = model.get_embed_params()
            proj_params = list(model.projector.parameters())
            stage1_params = head_params + embed_params + proj_params
            if stage1_params:
                freeze_epochs = ftcfg["freeze_epochs"]
                if has_rul and not has_cls:
                    freeze_epochs = max(freeze_epochs, 20)
                opt = AdamW(stage1_params, lr=ftcfg["lr_head"])
                sched = CosineAnnealingLR(opt, T_max=freeze_epochs)
                finetune_stage(
                    model, train_loader, val_loader,
                    cls_criterion, rul_criterion,
                    opt, sched, device, ds_idx, freq, n_ch,
                    has_cls, has_rul,
                    freeze_epochs, ftcfg["patience"], "Stage1-Frozen",
                )

            # ── Stage 2: Partial unfreeze ────────────────────────────
            # Unfreeze embeddings, projector, and last 2 encoder layers
            for p in model.get_embed_params():
                p.requires_grad = True

            encoder_layers = list(model.encoder.layers)
            for layer in encoder_layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True

            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = AdamW(trainable, lr=ftcfg["lr_backbone"])
            sched = CosineAnnealingLR(opt, T_max=ftcfg["partial_epochs"])
            finetune_stage(
                model, train_loader, val_loader,
                cls_criterion, rul_criterion,
                opt, sched, device, ds_idx, freq, n_ch,
                has_cls, has_rul,
                ftcfg["partial_epochs"], ftcfg["patience"],
                "Stage2-Partial",
            )

            # ── Stage 3: Full fine-tune ──────────────────────────────
            for p in model.parameters():
                p.requires_grad = True
            # Keep decoder frozen (only needed for pre-training)
            for p in model.get_decoder_params():
                p.requires_grad = False

            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = AdamW(trainable, lr=ftcfg["lr_backbone"])
            sched = CosineAnnealingLR(opt, T_max=ftcfg["full_epochs"])
            _, converge_ep = finetune_stage(
                model, train_loader, val_loader,
                cls_criterion, rul_criterion,
                opt, sched, device, ds_idx, freq, n_ch,
                has_cls, has_rul,
                ftcfg["full_epochs"], ftcfg["patience"], "Stage3-Full",
            )

        # ── Test evaluation ──────────────────────────────────────────
        test_results = evaluate_single(
            model, test_loader, device, ds_idx, freq, n_ch,
            has_cls, has_rul,
        )

        info = f"  -> Test:"
        if has_cls:
            info += (f" Acc={test_results.get('acc', 0):.4f}"
                     f" F1={test_results.get('f1', 0):.4f}")
        if has_rul:
            info += (f" MAE={test_results.get('rul_mae', 0):.4f}"
                     f" RMSE={test_results.get('rul_rmse', 0):.4f}")
        info += f"  |  Time: {timer.elapsed:.1f}s"
        print(info)

        logger.log({
            "dataset": name,
            "accuracy": round(test_results.get("acc", 0), 4),
            "f1_score": round(test_results.get("f1", 0), 4),
            "rul_mae": round(test_results.get("rul_mae", 0), 4),
            "rul_rmse": round(test_results.get("rul_rmse", 0), 4),
            "train_time_s": round(timer.elapsed, 1),
            "converge_epoch": converge_ep,
            "ft_stage": "3-stage",
        })

        torch.save(
            model.state_dict(),
            f"checkpoints/foundation_ft_{name}.pt",
        )

    logger.close()
    print(f"\n{'═' * 60}")
    print("Fine-tuning complete!")
    print("  Results -> results/foundation_metrics.csv")
    print(f"{'═' * 60}")


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Three-stage fine-tuning of pre-trained foundation model",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    fine_tune(args.config)
