# PHM Foundation Model

A **self-supervised foundation model** for Prognostics and Health Management (PHM) time-series data. Uses a **Masked Autoencoder (MAE)** with **Frequency-Conditioned LayerNorm** to learn general signal representations across 5 heterogeneous industrial domains, then transfers to fault classification and RUL prediction.

## Key Contributions

1. **Self-Supervised Pre-Training** -- Masked Autoencoder reconstructs randomly masked signal patches (no labels needed)
2. **Frequency-Conditioned LayerNorm (FreqCondNorm)** -- FiLM-style modulation adapts normalization parameters based on sampling frequency (1 Hz -- 97 kHz), enabling a single model to handle heterogeneous domains
3. **Cross-Domain Transfer** -- Pre-trained on 4 domains, achieves 82% zero-shot accuracy on held-out MFPT
4. **Few-Shot Learning** -- Outperforms baselines with as little as 1% labeled data

## Datasets (5 Domains)

| Dataset | Domain | Freq | Channels | Task | Classes |
|---------|--------|------|----------|------|---------|
| CWRU | Bearing vibration | 12 kHz | 1 | Classification | 4 |
| PRONOSTIA | Bearing degradation | 25.6 kHz | 2 | RUL regression | -- |
| CMAPSS | Turbofan engine | 1 Hz (cycle) | 14 | RUL regression | -- |
| MFPT | Bearing vibration | 97.6 kHz | 1 | Classification | 2 |
| UOC18 | Gear fault | 20 kHz | 1 | Classification | 9 |

## Results

### Full-Data Classification (Accuracy %)

| Dataset | Baseline CNN | Foundation Model | Delta |
|---------|-------------|-----------------|-------|
| CWRU | 92.8 | **99.2** | +6.4 |
| UOC18 | **100.0** | **100.0** | 0.0 |
| MFPT | **99.2** | 96.6 | -2.6 |

### Few-Shot Highlights
- **CWRU**: Foundation wins at >= 50% data (96.2% vs 93.1%)
- **MFPT**: Foundation wins at 1--10% data (75.5% vs 74.6% at 1%)
- **UOC18**: Foundation 2x better at 1% data (20.4% vs 10.3%)

### Cross-Domain (Leave-One-Out)
- MFPT zero-shot: **82.0%** accuracy without any fine-tuning
- CWRU linear probe: **73.7%** accuracy

## Architecture

```
Input (B, C, L)
    |
    +-- Patch Embedding: unfold -> Linear(patch_size=64, d_model=128)
    +-- Learnable Positional Encoding
    |
    +-- [PRE-TRAIN] Mask 40% of patches, encode ONLY visible tokens (true MAE)
    |
    +-- FreqConditioned Transformer Encoder (4 layers)
    |       FreqCondNorm: gamma, beta = MLP(log10(sampling_freq))
    |       Multi-Head Self-Attention (8 heads)
    |       FFN (GELU, dim=256)
    |
    +-- [PRE-TRAIN] Insert mask tokens -> MAE Decoder (2 layers) -> Reconstruct
    |       Loss: MSE on masked patches only (patch-level normalized targets)
    |
    +-- [FINE-TUNE] CLS pooling -> Dataset Embedding -> Projector -> Task Heads
            Classification: Linear(128, num_classes)
            RUL: MLP -> Hardtanh[0,1]
```

### 3-Stage Fine-Tuning
1. **Stage 1** (15 epochs): Train heads + projector + embeddings, backbone frozen
2. **Stage 2** (20 epochs): Unfreeze last 2 Transformer layers
3. **Stage 3** (30 epochs): Full fine-tune with low LR

## Project Structure

```
.
├── configs/
│   └── config.yaml              # Master configuration
├── scripts/
│   ├── run_phm.sh               # SLURM job script (HPC)
│   ├── generate_pretrain_plot.py # Plot pre-training loss curves
│   ├── make_architecture_pdf.py  # Generate architecture diagram
│   └── make_presentation.py      # Generate results presentation
├── results/
│   └── plots/                   # Generated plots (accuracy, t-SNE, etc.)
├── foundation_model.py          # MAE + FreqCondNorm architecture
├── data_pipeline.py             # Dataset loading, resampling, windowing, HDF5
├── pretrain.py                  # Self-supervised MAE pre-training loop
├── fine_tune.py                 # 3-stage transfer learning
├── evaluation.py                # Comparison, few-shot, cross-domain, t-SNE
├── baseline_model.py            # Per-dataset 1D CNN baseline
├── train_baseline.py            # Baseline training loop
├── ablation_studies.py          # Component ablation experiments
├── run_all.py                   # Master pipeline (7 steps)
├── utils.py                     # Seed, device, logging, metrics
├── presentation.tex             # LaTeX Beamer research presentation
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (data -> baselines -> pretrain -> finetune -> eval)
python run_all.py

# Run individual steps
python run_all.py --step 1   # Data loading & preprocessing
python run_all.py --step 2   # Baseline CNN training
python run_all.py --step 3   # Self-supervised pre-training (MAE)
python run_all.py --step 4   # 3-stage fine-tuning
python run_all.py --step 5   # Evaluation & plots
python run_all.py --step 6   # Ablation studies
python run_all.py --step 7   # Summary report

# Quick run (skip ablations)
python run_all.py --quick
```

### HPC (SLURM)

```bash
sbatch scripts/run_phm.sh
```

## Configuration

All hyperparameters are in `configs/config.yaml`:
- **Pre-training**: mask_ratio, decoder architecture, epochs, LR
- **Encoder**: d_model, patch_size, num_layers, FreqCondNorm
- **Fine-tuning**: stage epochs, learning rates, patience
- **Few-shot**: data fractions, number of seeds
- **Ablation**: sweep ranges
