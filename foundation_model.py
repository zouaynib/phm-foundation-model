"""
Foundation Model — Masked Autoencoder with Frequency-Conditioned Normalization
===============================================================================

A self-supervised pre-training framework for multi-domain industrial
time-series, combining three key ideas:

  1. **PatchTST-style channel-independent processing**: each sensor channel
     passes through the SAME Transformer encoder independently, then outputs
     are averaged.  This naturally handles variable channel counts.

  2. **Masked Autoencoder (MAE) pre-training**: randomly mask a fraction of
     input patches and train the model to reconstruct them.  The encoder
     learns general signal representations without any labels.

  3. **Frequency-Conditioned Layer Normalization (FreqCondNorm)**: replaces
     standard LayerNorm with FiLM-style modulation where the affine
     parameters (gamma, beta) are generated from log10(sampling_frequency).
     This allows a single model to process signals from 1 Hz to 200 kHz.

Architecture Overview
---------------------
ENCODER (shared across pre-training and fine-tuning):
    PatchEmbedding -> Positional Encoding -> FreqCondTransformerEncoder

PRE-TRAINING PATH (self-supervised):
    Random masking -> Encoder (masked tokens) -> Decoder -> Reconstruction
    Loss: MSE on masked patches only

FINE-TUNING PATH (supervised):
    Encoder (no masking) -> Channel pool -> Dataset embed -> Projector -> Heads

References
----------
- PatchTST: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023
- MAE: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- FiLM: Perez et al., "FiLM: Visual Reasoning with General Conditioning", AAAI 2018
"""

import math
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
# 1. FREQUENCY-CONDITIONED LAYER NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

class FreqConditionedLayerNorm(nn.Module):
    """Layer normalization with FiLM-style frequency conditioning.

    Standard LayerNorm learns fixed affine parameters (gamma, beta).
    FreqCondNorm instead *generates* gamma and beta from the input signal's
    sampling frequency via a small MLP.  This allows the normalization to
    adapt to the frequency content of the signal — a 12 kHz vibration signal
    and a 200 kHz acoustic emission signal get different normalizations
    even if their raw amplitudes are similar.

    Initialisation is close to identity (gamma ≈ 1, beta ≈ 0) so that
    the model starts behaving like standard LayerNorm and gradually learns
    frequency-specific adjustments.

    Parameters
    ----------
    d_model : int
        Feature dimension to normalize.
    freq_dim : int
        Hidden dimension of the gamma/beta generator MLP.
    """

    def __init__(self, d_model: int, freq_dim: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Gamma generator: log10(fs) -> gamma (multiplicative modulation)
        self.gamma_net = nn.Sequential(
            nn.Linear(1, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, d_model),
        )
        # Beta generator: log10(fs) -> beta (additive modulation)
        self.beta_net = nn.Sequential(
            nn.Linear(1, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, d_model),
        )

        # Initialize close to identity: gamma ≈ 1, beta ≈ 0
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x: torch.Tensor, log_freq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
            Input features (e.g. patch tokens).
        log_freq : Tensor, shape (B, 1)
            log10 of sampling frequency for each sample.

        Returns
        -------
        Tensor, shape (B, T, D)
            Frequency-conditioned normalized features.
        """
        normed = self.norm(x)
        gamma = self.gamma_net(log_freq).unsqueeze(1)   # (B, 1, D)
        beta = self.beta_net(log_freq).unsqueeze(1)      # (B, 1, D)
        return gamma * normed + beta


# ═══════════════════════════════════════════════════════════════════════
# 2. FREQUENCY-CONDITIONED TRANSFORMER LAYER
# ═══════════════════════════════════════════════════════════════════════

class FreqCondTransformerLayer(nn.Module):
    """Pre-norm Transformer encoder layer with FreqCondNorm.

    Replaces both LayerNorms in the standard Transformer layer with
    FreqConditionedLayerNorm, allowing self-attention and FFN sub-layers
    to adapt their behaviour based on the signal's sampling frequency.

    Uses pre-norm (norm before attention/FFN) for training stability,
    following modern Transformer best practices.

    Parameters
    ----------
    d_model : int
        Model dimension.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        FFN intermediate dimension.
    dropout : float
        Dropout probability.
    freq_dim : int
        Hidden dimension for FreqCondNorm MLP.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, freq_dim: int = 32):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )

        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Frequency-conditioned normalizations (pre-norm style)
        self.norm1 = FreqConditionedLayerNorm(d_model, freq_dim)
        self.norm2 = FreqConditionedLayerNorm(d_model, freq_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, log_freq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
            Input token sequence.
        log_freq : Tensor, shape (B, 1)
            log10(sampling_frequency) for frequency conditioning.

        Returns
        -------
        Tensor, shape (B, T, D)
        """
        # Pre-norm self-attention
        normed = self.norm1(x, log_freq)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + self.dropout1(attn_out)

        # Pre-norm feed-forward
        normed = self.norm2(x, log_freq)
        ff_out = self.ffn(normed)
        x = x + self.dropout2(ff_out)

        return x


class FreqCondTransformerEncoder(nn.Module):
    """Stack of FreqCondTransformerLayers.

    Parameters
    ----------
    d_model, nhead, num_layers, dim_feedforward, dropout, freq_dim :
        Forwarded to each FreqCondTransformerLayer.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, freq_dim: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            FreqCondTransformerLayer(d_model, nhead, dim_feedforward,
                                    dropout, freq_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, log_freq: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, log_freq)
        return x


# ═══════════════════════════════════════════════════════════════════════
# 3. PATCH EMBEDDING & POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Unfold a 1-D signal into overlapping patches and project each
    patch to d_model dimensions via a linear layer.

    Parameters
    ----------
    patch_size : int
        Number of time-steps per patch.
    patch_stride : int
        Stride between consecutive patches.
    d_model : int
        Output embedding dimension.
    """

    def __init__(self, patch_size: int, patch_stride: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, L)
            Single-channel waveform.

        Returns
        -------
        Tensor, shape (B, num_patches, d_model)
        """
        # unfold(dim, size, step) -> (B, num_patches, patch_size)
        patches = x.unfold(1, self.patch_size, self.patch_stride)
        return self.proj(patches)

    def get_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw patches without projection (for reconstruction targets).

        Returns
        -------
        Tensor, shape (B, num_patches, patch_size)
        """
        return x.unfold(1, self.patch_size, self.patch_stride)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding added to patch tokens.

    Parameters
    ----------
    max_patches : int
        Maximum number of patches (determines embedding table size).
    d_model : int
        Embedding dimension.
    """

    def __init__(self, max_patches: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_patches, d_model) -> (B, num_patches, d_model)"""
        return x + self.pos_embed[:, :x.shape[1], :]


# ═══════════════════════════════════════════════════════════════════════
# 4. MAE DECODER
# ═══════════════════════════════════════════════════════════════════════

class MAEDecoder(nn.Module):
    """Lightweight Transformer decoder for masked patch reconstruction.

    Uses standard LayerNorm (not frequency-conditioned) because:
    1. The encoder has already processed frequency information.
    2. Keeping the decoder simple follows the MAE design principle.
    3. Reduces parameter count (decoder should be much smaller than encoder).

    Parameters
    ----------
    encoder_d_model : int
        Encoder output dimension (input to decoder via linear projection).
    d_model : int
        Decoder hidden dimension.
    num_heads : int
        Decoder attention heads.
    num_layers : int
        Number of decoder Transformer layers.
    dim_feedforward : int
        Decoder FFN intermediate dimension.
    dropout : float
        Dropout probability.
    patch_size : int
        Number of time-steps per patch (output dimension for reconstruction).
    """

    def __init__(self, encoder_d_model: int, d_model: int, num_heads: int,
                 num_layers: int, dim_feedforward: int, dropout: float,
                 patch_size: int):
        super().__init__()

        # Project encoder output to decoder dimension
        self.input_proj = nn.Linear(encoder_d_model, d_model)

        # Decoder positional encoding (learned, same as encoder)
        # We use a generous max_patches; actual slicing happens in forward
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        # Standard Transformer encoder layers (used as decoder)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)

        # Reconstruct original patch values
        self.reconstruct = nn.Linear(d_model, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, num_patches, encoder_d_model)
            Full sequence of encoded tokens (visible + mask tokens).

        Returns
        -------
        Tensor, shape (B, num_patches, patch_size)
            Reconstructed patches.
        """
        x = self.input_proj(x)                               # (B, N, d_dec)
        x = x + self.pos_embed[:, :x.shape[1], :]            # add position
        x = self.transformer(x)                               # decode
        x = self.norm(x)
        return self.reconstruct(x)                            # (B, N, patch_size)


# ═══════════════════════════════════════════════════════════════════════
# 5. FOUNDATION MODEL
# ═══════════════════════════════════════════════════════════════════════

class FoundationModel(nn.Module):
    """Multi-domain foundation model for PHM time-series.

    Supports two operating modes:

    **Pre-training** (``forward_pretrain``):
        Masked autoencoder — randomly mask patches, encode, decode,
        reconstruct masked patches.  No labels needed.

    **Fine-tuning** (``forward`` / ``forward_single_dataset``):
        Encode all patches, pool, project, route to per-dataset
        classification and/or RUL regression heads.

    Parameters
    ----------
    dataset_configs : list of dict
        Per-dataset configuration.  Each dict has keys:
        - 'num_channels': int
        - 'tasks': list of {'type': 'classification'|'regression',
                            'num_classes': int (for classification)}
    window_length : int
        Signal length L after windowing/padding.
    d_model : int
        Transformer encoder hidden dimension.
    patch_size, patch_stride : int
        Patching parameters.
    num_heads, num_layers : int
        Transformer encoder depth and width.
    dim_feedforward : int
        FFN intermediate dimension.
    dropout : float
        Dropout rate.
    freq_dim : int
        Hidden dimension for FreqCondNorm MLP.
    dataset_embed_dim : int
        Dataset ID embedding dimension.
    latent_dim : int
        Output dimension after projector MLP.
    max_channels : int
        Maximum number of channels across all datasets.
    decoder_d_model, decoder_num_heads, decoder_num_layers,
    decoder_dim_feedforward : int
        MAE decoder configuration.
    use_freq_cond : bool
        Whether to use frequency-conditioned normalization.
        When False, uses standard LayerNorm (for ablation).
    use_dataset_embed : bool
        Whether to use dataset ID embedding (for ablation).
    """

    def __init__(
        self,
        dataset_configs,
        window_length: int = 2560,
        d_model: int = 128,
        patch_size: int = 64,
        patch_stride: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        freq_dim: int = 32,
        dataset_embed_dim: int = 32,
        latent_dim: int = 128,
        max_channels: int = 14,
        decoder_d_model: int = 64,
        decoder_num_heads: int = 4,
        decoder_num_layers: int = 2,
        decoder_dim_feedforward: int = 128,
        use_freq_cond: bool = True,
        use_dataset_embed: bool = True,
    ):
        super().__init__()

        # Store config
        self.d_model = d_model
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.max_channels = max_channels
        self.num_datasets = len(dataset_configs)
        self.dataset_configs = dataset_configs
        self.use_freq_cond = use_freq_cond
        self.use_dataset_embed = use_dataset_embed

        # ── Patch Embedding ──────────────────────────────────────────
        self.patch_embed = PatchEmbedding(patch_size, patch_stride, d_model)
        num_patches = (window_length - patch_size) // patch_stride + 1
        self.num_patches = num_patches
        self.pos_encode = LearnablePositionalEncoding(num_patches, d_model)

        # ── Encoder ──────────────────────────────────────────────────
        if use_freq_cond:
            # Frequency-conditioned Transformer encoder (our contribution)
            self.encoder = FreqCondTransformerEncoder(
                d_model, num_heads, num_layers, dim_feedforward,
                dropout, freq_dim,
            )
            self.encoder_norm = FreqConditionedLayerNorm(d_model, freq_dim)
        else:
            # Standard Transformer encoder (for ablation comparison)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )
            self.encoder_norm = nn.LayerNorm(d_model)

        # ── Pre-training: Mask Token + Decoder ───────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = MAEDecoder(
            encoder_d_model=d_model,
            d_model=decoder_d_model,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            dim_feedforward=decoder_dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
        )

        # ── Fine-tuning: Dataset Embedding + Projector + Heads ───────
        feat_dim = d_model
        if use_dataset_embed:
            self.ds_embed = nn.Embedding(self.num_datasets, dataset_embed_dim)
            feat_dim += dataset_embed_dim

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-dataset task heads
        self.cls_heads = nn.ModuleDict()
        self.rul_heads = nn.ModuleDict()

        for ds_id, ds_cfg in enumerate(dataset_configs):
            for task in ds_cfg.get("tasks", []):
                if task["type"] == "classification":
                    num_classes = task["num_classes"]
                    self.cls_heads[f"cls_{ds_id}"] = nn.Linear(
                        latent_dim, num_classes,
                    )
                elif task["type"] == "regression":
                    self.rul_heads[f"rul_{ds_id}"] = nn.Sequential(
                        nn.Linear(latent_dim, latent_dim),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(latent_dim, latent_dim // 2),
                        nn.GELU(),
                        nn.Linear(latent_dim // 2, 1),
                        nn.Hardtanh(min_val=0.0, max_val=1.0),
                    )

    # ──────────────────────────────────────────────────────────────────
    # ENCODER BACKBONE
    # ──────────────────────────────────────────────────────────────────

    def _encode_tokens(
        self, tokens: torch.Tensor, log_freq: torch.Tensor,
    ) -> torch.Tensor:
        """Run tokens through the Transformer encoder + final norm.

        Parameters
        ----------
        tokens : Tensor, shape (B, N, d_model)
            Patch token sequence (possibly with mask tokens).
        log_freq : Tensor, shape (B, 1)
            log10(sampling_frequency).

        Returns
        -------
        Tensor, shape (B, N, d_model)
            Encoded token sequence.
        """
        if self.use_freq_cond:
            tokens = self.encoder(tokens, log_freq)
            tokens = self.encoder_norm(tokens, log_freq)
        else:
            tokens = self.encoder(tokens)
            tokens = self.encoder_norm(tokens)
        return tokens

    def forward_backbone(
        self, x: torch.Tensor, freq: torch.Tensor,
        num_channels: torch.Tensor,
    ) -> torch.Tensor:
        """Channel-independent Transformer backbone.

        Each channel is processed independently through the same encoder,
        then outputs are averaged across channels.

        Parameters
        ----------
        x : Tensor, shape (B, C_max, L)
            Zero-padded multivariate signal.
        freq : Tensor, shape (B,)
            Sampling frequency per sample.
        num_channels : Tensor, shape (B,)
            Actual number of channels per sample.

        Returns
        -------
        Tensor, shape (B, d_model)
            Aggregated representation.
        """
        B, C_max, L = x.shape
        device = x.device
        log_freq = torch.log10(freq.clamp(min=1e-3)).unsqueeze(-1)  # (B, 1)

        unique_nch = torch.unique(num_channels)

        # ── Fast path: all samples have the same number of channels ──
        if unique_nch.numel() == 1:
            C = unique_nch.item()
            xc = x[:, :C, :].reshape(B * C, L)             # (B*C, L)

            tokens = self.patch_embed(xc)                    # (B*C, N, d)
            tokens = self.pos_encode(tokens)

            # Expand log_freq for channel-independent processing
            lf = log_freq.repeat_interleave(C, dim=0)        # (B*C, 1)

            tokens = self._encode_tokens(tokens, lf)
            pooled = tokens.mean(dim=1)                       # (B*C, d)
            pooled = pooled.view(B, C, self.d_model).mean(1)  # (B, d)
            return pooled

        # ── Slow path: mixed channel counts ──────────────────────────
        out = torch.zeros(B, self.d_model, device=device)

        for nch in unique_nch:
            nch_int = nch.item()
            mask = num_channels == nch
            idx = mask.nonzero(as_tuple=True)[0]
            B_sub = idx.shape[0]

            xc = x[idx, :nch_int, :].reshape(B_sub * nch_int, L)
            tokens = self.patch_embed(xc)
            tokens = self.pos_encode(tokens)

            lf = log_freq[idx].repeat_interleave(nch_int, dim=0)
            tokens = self._encode_tokens(tokens, lf)

            pooled = tokens.mean(dim=1)
            pooled = pooled.view(B_sub, nch_int, self.d_model).mean(1)
            out[idx] = pooled

        return out

    # ──────────────────────────────────────────────────────────────────
    # PRE-TRAINING: MASKED AUTOENCODER
    # ──────────────────────────────────────────────────────────────────

    def forward_pretrain(
        self, x: torch.Tensor, freq: torch.Tensor,
        num_channels: torch.Tensor, mask_ratio: float = 0.4,
    ) -> tuple:
        """True MAE-style self-supervised masked patch reconstruction.

        Following He et al. (CVPR 2022):
        1. Split signal into patches, embed them.
        2. Randomly mask ``mask_ratio`` fraction of patches.
        3. Encoder processes ONLY visible (unmasked) tokens.
        4. After encoding, insert learnable [MASK] tokens at masked positions.
        5. Decoder reconstructs all patches; loss computed only on masked ones.
        6. Reconstruction targets are patch-normalized (zero mean, unit var).

        Parameters
        ----------
        x : Tensor, shape (B, C_max, L)
        freq : Tensor, shape (B,)
        num_channels : Tensor, shape (B,)
        mask_ratio : float

        Returns
        -------
        loss : Tensor, scalar
        details : dict
        """
        B, C_max, L = x.shape
        device = x.device
        log_freq = torch.log10(freq.clamp(min=1e-3)).unsqueeze(-1)

        unique_nch = torch.unique(num_channels)
        total_loss = torch.tensor(0.0, device=device)
        total_masked = 0
        total_patches = 0

        for nch in unique_nch:
            nch_int = nch.item()
            mask_batch = num_channels == nch
            idx = mask_batch.nonzero(as_tuple=True)[0]
            B_sub = idx.shape[0]

            xc = x[idx, :nch_int, :]                     # (B_sub, C, L)
            xc_flat = xc.reshape(B_sub * nch_int, L)     # (B_sub*C, L)

            # Get raw patches (reconstruction targets)
            raw_patches = self.patch_embed.get_patches(xc_flat)  # (B_sub*C, N, ps)
            N = raw_patches.shape[1]

            # ── Patch-level normalization (per MAE paper) ──────────
            # Normalize each patch to zero mean, unit variance.
            # This forces the model to learn structure, not amplitude.
            patch_mean = raw_patches.mean(dim=-1, keepdim=True)
            patch_var = raw_patches.var(dim=-1, keepdim=True)
            target_patches = (raw_patches - patch_mean) / (patch_var + 1e-6).sqrt()

            # Embed patches + positional encoding
            tokens = self.patch_embed(xc_flat)            # (B_sub*C, N, d)
            tokens = self.pos_encode(tokens)

            # ── Generate mask (same for all channels of a sample) ──
            num_mask = max(1, int(N * mask_ratio))
            num_visible = N - num_mask
            noise = torch.rand(B_sub, N, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Visible token indices (first num_visible in shuffled order)
            ids_keep = ids_shuffle[:, :num_visible]       # (B_sub, num_visible)

            # Expand for all channels
            ids_keep_exp = ids_keep.repeat_interleave(nch_int, dim=0)  # (B_sub*C, num_visible)
            ids_restore_exp = ids_restore.repeat_interleave(nch_int, dim=0)  # (B_sub*C, N)

            # ── TRUE MAE: encode ONLY visible tokens ──────────────
            visible_tokens = torch.gather(
                tokens, 1,
                ids_keep_exp.unsqueeze(-1).expand(-1, -1, self.d_model),
            )  # (B_sub*C, num_visible, d)

            lf = log_freq[idx].repeat_interleave(nch_int, dim=0)
            encoded_visible = self._encode_tokens(visible_tokens, lf)  # (B_sub*C, num_visible, d)

            # ── Insert mask tokens at masked positions ─────────────
            BC = B_sub * nch_int
            mask_tokens = self.mask_token.expand(BC, num_mask, -1)  # (BC, num_mask, d)
            # Concatenate: [encoded_visible, mask_tokens]
            full_tokens = torch.cat([encoded_visible, mask_tokens], dim=1)  # (BC, N, d)
            # Un-shuffle to restore original patch order
            full_tokens = torch.gather(
                full_tokens, 1,
                ids_restore_exp.unsqueeze(-1).expand(-1, -1, self.d_model),
            )  # (BC, N, d)

            # ── Decode ─────────────────────────────────────────────
            reconstructed = self.decoder(full_tokens)     # (BC, N, ps)

            # ── MSE loss on masked patches only ────────────────────
            # Build boolean mask
            mask_per_sample = torch.zeros(B_sub, N, device=device)
            mask_per_sample.scatter_(
                1, ids_shuffle[:, num_visible:],
                torch.ones(B_sub, num_mask, device=device),
            )
            mask_expanded = mask_per_sample.repeat_interleave(nch_int, dim=0)
            mask_bool = mask_expanded.bool()

            pred_masked = reconstructed[mask_bool]        # (M, ps)
            target_masked = target_patches[mask_bool]     # (M, ps)

            if pred_masked.numel() > 0:
                loss_sub = nn.functional.mse_loss(
                    pred_masked, target_masked, reduction='sum',
                )
                total_loss = total_loss + loss_sub
                total_masked += pred_masked.shape[0]

            total_patches += B_sub * nch_int * N

        loss = total_loss / max(total_masked, 1)

        details = {
            'num_masked': total_masked,
            'num_total': total_patches,
            'loss_per_patch': loss.item(),
        }
        return loss, details

    # ──────────────────────────────────────────────────────────────────
    # FINE-TUNING: SUPERVISED FORWARD
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, freq: torch.Tensor,
        dataset_id: torch.Tensor, num_channels: torch.Tensor,
    ):
        """Supervised forward pass for fine-tuning.

        Parameters
        ----------
        x : Tensor, shape (B, C_max, L)
        freq : Tensor, shape (B,)
        dataset_id : Tensor, shape (B,)
        num_channels : Tensor, shape (B,)

        Returns
        -------
        cls_outputs : dict {ds_id_int: Tensor (N_ds, num_classes)}
        rul_outputs : dict {ds_id_int: Tensor (N_ds,)}
        latent : Tensor, shape (B, latent_dim)
        """
        backbone_feat = self.forward_backbone(x, freq, num_channels)

        # Build feature vector: backbone + optional dataset embedding
        parts = [backbone_feat]
        if self.use_dataset_embed:
            parts.append(self.ds_embed(dataset_id))

        feat = torch.cat(parts, dim=-1)
        latent = self.projector(feat)

        # Route to per-dataset task heads
        unique_ds = torch.unique(dataset_id)
        cls_outputs = {}
        rul_outputs = {}

        for ds_id in unique_ds:
            ds_id_int = ds_id.item()
            ds_mask = dataset_id == ds_id
            ds_latent = latent[ds_mask]

            cls_key = f"cls_{ds_id_int}"
            if cls_key in self.cls_heads:
                cls_outputs[ds_id_int] = self.cls_heads[cls_key](ds_latent)

            rul_key = f"rul_{ds_id_int}"
            if rul_key in self.rul_heads:
                rul_outputs[ds_id_int] = self.rul_heads[rul_key](
                    ds_latent,
                ).squeeze(-1)

        return cls_outputs, rul_outputs, latent

    def forward_single_dataset(
        self, x: torch.Tensor, freq: float,
        dataset_id_int: int, num_channels_int: int,
    ):
        """Convenience method when all samples come from one dataset.

        Returns
        -------
        cls_logits : Tensor or None
        rul_preds : Tensor or None
        """
        B = x.shape[0]
        device = x.device
        freq_t = (
            freq if isinstance(freq, torch.Tensor)
            else torch.full((B,), freq, device=device)
        )
        dsid_t = torch.full((B,), dataset_id_int, dtype=torch.long,
                            device=device)
        nch_t = torch.full((B,), num_channels_int, dtype=torch.long,
                           device=device)

        cls_outputs, rul_outputs, latent = self(x, freq_t, dsid_t, nch_t)
        return cls_outputs.get(dataset_id_int), rul_outputs.get(dataset_id_int)

    # ──────────────────────────────────────────────────────────────────
    # REPRESENTATION EXTRACTION (for evaluation / t-SNE)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_representations(
        self, x: torch.Tensor, freq: torch.Tensor,
        num_channels: torch.Tensor,
    ) -> torch.Tensor:
        """Extract backbone features without task heads.

        Useful for t-SNE visualization and linear probing.

        Returns
        -------
        Tensor, shape (B, d_model)
        """
        self.eval()
        return self.forward_backbone(x, freq, num_channels)

    # ──────────────────────────────────────────────────────────────────
    # PARAMETER GROUPS (for staged fine-tuning)
    # ──────────────────────────────────────────────────────────────────

    def get_backbone_params(self):
        """Encoder + patch embedding + positional encoding parameters."""
        params = []
        params.extend(self.patch_embed.parameters())
        params.extend(self.pos_encode.parameters())
        params.extend(self.encoder.parameters())
        if isinstance(self.encoder_norm, nn.Module):
            params.extend(self.encoder_norm.parameters())
        return params

    def get_decoder_params(self):
        """MAE decoder + mask token parameters."""
        params = list(self.decoder.parameters())
        params.append(self.mask_token)
        return params

    def get_head_params(self, ds_id=None):
        """Classification + RUL head parameters."""
        params = []
        if ds_id is not None:
            cls_key = f"cls_{ds_id}"
            rul_key = f"rul_{ds_id}"
            if cls_key in self.cls_heads:
                params.extend(self.cls_heads[cls_key].parameters())
            if rul_key in self.rul_heads:
                params.extend(self.rul_heads[rul_key].parameters())
        else:
            for head in self.cls_heads.values():
                params.extend(head.parameters())
            for head in self.rul_heads.values():
                params.extend(head.parameters())
        return params

    def get_embed_params(self):
        """Dataset embedding + projector parameters."""
        params = list(self.projector.parameters())
        if self.use_dataset_embed:
            params.extend(self.ds_embed.parameters())
        return params


# ═══════════════════════════════════════════════════════════════════════
# 6. SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Dataset configs matching 5 domains
    ds_configs = [
        {"num_channels": 1,
         "tasks": [{"type": "classification", "num_classes": 4}]},
        {"num_channels": 2,
         "tasks": [{"type": "regression"}]},
        {"num_channels": 14,
         "tasks": [{"type": "classification", "num_classes": 2},
                   {"type": "regression"}]},
        {"num_channels": 1,
         "tasks": [{"type": "classification", "num_classes": 3}]},
        {"num_channels": 1,
         "tasks": [{"type": "classification", "num_classes": 3}]},
    ]

    window_length = 2560
    max_ch = 14

    model = FoundationModel(
        ds_configs, window_length=window_length, max_channels=max_ch,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # ── Test pre-training forward ──
    print("\n--- Pre-training forward test ---")
    B = 8
    x = torch.randn(B, max_ch, window_length)
    freq = torch.full((B,), 25600.0)
    nch = torch.full((B,), 2, dtype=torch.long)

    loss, details = model.forward_pretrain(x, freq, nch, mask_ratio=0.4)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Masked patches: {details['num_masked']} / {details['num_total']}")

    # ── Test fine-tuning forward ──
    print("\n--- Fine-tuning forward test ---")
    dsid = torch.zeros(B, dtype=torch.long)
    cls_out, rul_out, latent = model(x, freq, dsid, nch)
    print(f"  Latent: {latent.shape}")
    for k, v in cls_out.items():
        print(f"  cls[{k}]: {v.shape}")
    for k, v in rul_out.items():
        print(f"  rul[{k}]: {v.shape}")

    # ── Test representation extraction ──
    print("\n--- Representation extraction test ---")
    reps = model.extract_representations(x, freq, nch)
    print(f"  Representations: {reps.shape}")

    # ── Parameter groups ──
    print("\n--- Parameter group counts ---")
    n_backbone = sum(p.numel() for p in model.get_backbone_params())
    n_decoder = sum(p.numel() for p in model.get_decoder_params())
    n_heads = sum(p.numel() for p in model.get_head_params())
    n_embeds = sum(p.numel() for p in model.get_embed_params())
    print(f"  Backbone (encoder+patch+pos+norm): {n_backbone:,}")
    print(f"  Decoder (MAE):                     {n_decoder:,}")
    print(f"  Heads (cls+rul):                   {n_heads:,}")
    print(f"  Embeddings (ds_embed+projector):   {n_embeds:,}")
    print(f"  Total:                             {n_params:,}")

    print("\nAll tests passed.")
