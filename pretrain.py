"""
Self-Supervised Pre-Training — Masked Autoencoder for PHM Time-Series
=====================================================================

Trains the foundation model encoder via masked patch reconstruction:
  1. Randomly mask 40% of input patches (configurable).
  2. Encode the full sequence (masked tokens replaced with [MASK]).
  3. Decode and reconstruct masked patches.
  4. Loss = MSE on masked patches only.

No labels are used.  The model learns general signal structure from
raw waveforms across all industrial domains jointly.

Key training features:
  - Balanced domain sampling (WeightedRandomSampler)
  - Linear LR warmup + cosine annealing
  - Mixed-precision training (FP16 on CUDA)
  - Gradient clipping
  - Early stopping on validation reconstruction loss

Usage:
    python pretrain.py
    python pretrain.py --config configs/config.yaml
"""

import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import (
    load_config, set_seed, get_device, ensure_dirs,
    CSVLogger, EarlyStopping, Timer,
)
from data_pipeline import PHMDataset, get_all_split_indices
from foundation_model import FoundationModel
import h5py


# ─────────────────────────────────────────────────────────────────────
# Balanced Domain Sampler
# ─────────────────────────────────────────────────────────────────────

def build_balanced_sampler(hdf5_path: str, indices: np.ndarray):
    """Weight each sample inversely proportional to its dataset size.

    This ensures each domain contributes equally to each epoch,
    preventing large datasets from dominating pre-training.

    Parameters
    ----------
    hdf5_path : str
        Path to combined HDF5 file.
    indices : ndarray
        Training sample indices.

    Returns
    -------
    WeightedRandomSampler
    """
    with h5py.File(hdf5_path, "r") as f:
        dsids = f["dataset_id"][:][indices]
    unique, counts = np.unique(dsids, return_counts=True)
    weight_map = {uid: 1.0 / c for uid, c in zip(unique, counts)}
    weights = np.array([weight_map[d] for d in dsids])
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_pretrain(model, loader, device, mask_ratio):
    """Compute average reconstruction loss on the validation set.

    Parameters
    ----------
    model : FoundationModel
    loader : DataLoader
    device : torch.device
    mask_ratio : float

    Returns
    -------
    float
        Average reconstruction MSE.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for sigs, _lbls, _rul, freqs, _dsids, nchannels in loader:
        sigs = sigs.to(device)
        freqs = freqs.to(device, dtype=torch.float32)
        nchannels = nchannels.to(device)

        loss, _ = model.forward_pretrain(sigs, freqs, nchannels, mask_ratio)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, scaler, use_amp,
                    mask_ratio, grad_clip):
    """Run one epoch of masked autoencoder pre-training.

    Parameters
    ----------
    model : FoundationModel
    loader : DataLoader
    optimizer : AdamW
    device : torch.device
    scaler : GradScaler or None
    use_amp : bool
    mask_ratio : float
    grad_clip : float

    Returns
    -------
    dict
        Training metrics: 'loss', 'num_masked', 'num_total'.
    """
    model.train()
    total_loss = 0.0
    total_masked = 0
    total_patches = 0
    n_batches = 0

    for sigs, _lbls, _rul, freqs, _dsids, nchannels in loader:
        sigs = sigs.to(device)
        freqs = freqs.to(device, dtype=torch.float32)
        nchannels = nchannels.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss, details = model.forward_pretrain(
                    sigs, freqs, nchannels, mask_ratio,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, details = model.forward_pretrain(
                sigs, freqs, nchannels, mask_ratio,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_masked += details['num_masked']
        total_patches += details['num_total']
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'num_masked': total_masked,
        'num_total': total_patches,
    }


# ─────────────────────────────────────────────────────────────────────
# Main Pre-Training Function
# ─────────────────────────────────────────────────────────────────────

def pretrain(config_path="configs/config.yaml"):
    """Run self-supervised masked autoencoder pre-training.

    Saves the best encoder checkpoint (by validation loss) to
    ``checkpoints/pretrained_encoder.pt``.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Best model state dict.
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    ensure_dirs("results", "checkpoints")

    hdf5_path = cfg["data"]["combined_hdf5"]
    pcfg = cfg["pretrain"]
    fcfg = cfg["foundation"]
    num_ds = len(cfg["datasets"])
    L = cfg["data"]["window_length"]
    use_amp = pcfg["use_mixed_precision"] and device.type == "cuda"
    mask_ratio = pcfg["mask_ratio"]

    # ── Data ─────────────────────────────────────────────────────────
    tr_idx, va_idx, _ = get_all_split_indices(
        hdf5_path, num_ds,
        cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["seed"],
    )

    sampler = build_balanced_sampler(hdf5_path, tr_idx)
    train_ds = PHMDataset(hdf5_path, indices=tr_idx)
    train_loader = DataLoader(
        train_ds, batch_size=pcfg["batch_size"],
        sampler=sampler, num_workers=0, pin_memory=True,
    )
    val_ds = PHMDataset(hdf5_path, indices=va_idx)
    val_loader = DataLoader(
        val_ds, batch_size=pcfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────
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

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.get_backbone_params())
    n_decoder = sum(p.numel() for p in model.get_decoder_params())
    print(f"Foundation Model: {n_params:,} total parameters")
    print(f"  Encoder: {n_encoder:,}  |  Decoder: {n_decoder:,}")
    print(f"  Device: {device}  |  AMP: {use_amp}  |  Mask ratio: {mask_ratio}")

    # ── Optimizer & Scheduler ────────────────────────────────────────
    # Only train encoder + decoder (not task heads)
    pretrain_params = (
        model.get_backbone_params()
        + model.get_decoder_params()
    )
    optimizer = AdamW(
        pretrain_params,
        lr=pcfg["lr"], weight_decay=pcfg["weight_decay"],
    )

    warmup_epochs = pcfg.get("warmup_epochs", 10)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, pcfg["epochs"] - warmup_epochs),
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs],
    )

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    early_stop = EarlyStopping(patience=pcfg["patience"])

    # ── Logging ──────────────────────────────────────────────────────
    logger = CSVLogger(
        "results/pretrain_log.csv",
        ["epoch", "train_loss", "val_loss", "lr"],
    )

    best_val_loss = float("inf")
    best_state = None
    converge_epoch = 0

    # ── Training ─────────────────────────────────────────────────────
    print(f"\nStarting self-supervised pre-training for {pcfg['epochs']} epochs...")
    print(f"{'─' * 60}")

    with Timer() as timer:
        for epoch in range(1, pcfg["epochs"] + 1):
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, scaler, use_amp,
                mask_ratio, pcfg.get("grad_clip", 1.0),
            )

            # Validate
            val_loss = evaluate_pretrain(
                model, val_loader, device, mask_ratio,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            # Log
            logger.log({
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "val_loss": round(val_loss, 6),
                "lr": round(current_lr, 8),
            })

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                converge_epoch = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{pcfg['epochs']}  |  "
                    f"train_loss={train_metrics['loss']:.5f}  |  "
                    f"val_loss={val_loss:.5f}  |  "
                    f"lr={current_lr:.2e}  |  "
                    f"best={best_val_loss:.5f} @ ep {converge_epoch}"
                )

            # Early stopping (minimize loss -> negate for EarlyStopping)
            if early_stop.step(-val_loss):
                print(f"  Early stopping at epoch {epoch}")
                break

    logger.close()

    # ── Save ─────────────────────────────────────────────────────────
    torch.save(best_state, "checkpoints/pretrained_encoder.pt")
    print(f"\n{'═' * 60}")
    print(f"Self-supervised pre-training complete!")
    print(f"  Best val loss: {best_val_loss:.5f} @ epoch {converge_epoch}")
    print(f"  Training time: {timer.elapsed:.1f}s ({timer.elapsed/60:.1f} min)")
    print(f"  Saved -> checkpoints/pretrained_encoder.pt")
    print(f"  Log   -> results/pretrain_log.csv")
    print(f"{'═' * 60}")

    return best_state


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-supervised pre-training (masked autoencoder)",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    pretrain(args.config)
