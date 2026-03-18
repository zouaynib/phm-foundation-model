"""
Generate a clean PDF with all architecture Q&A for the PHM Foundation Model.
"""

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "PHM Foundation Model - Architecture Reference", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(25, 60, 120)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 60, 120)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def question(self, text):
        self.ln(2)
        self.set_font("Helvetica", "BI", 11)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 6, f"Q: {text}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def answer(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, chr(8226))
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        kw = self.get_string_width(key + ":  ") + 2
        self.cell(kw, 5.5, f"{key}:")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, f" {value}", new_x="LMARGIN", new_y="NEXT")


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ── TITLE PAGE ──
pdf.ln(30)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 14, "PHM Foundation Model", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Architecture & Design Reference", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "Zaynab Raounak", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)

pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(50, 50, 50)
pdf.multi_cell(0, 6.5,
    "This document provides a comprehensive reference for the architecture, design decisions, "
    "training strategy, and data pipeline of a multi-domain foundation model for Prognostics & "
    "Health Management (PHM). It covers all components from input preprocessing to task-specific "
    "heads, and includes answers to anticipated technical questions.",
    align="C", new_x="LMARGIN", new_y="NEXT")

# ── 1. HIGH-LEVEL ARCHITECTURE ──
pdf.add_page()
pdf.section_title("1. High-Level Architecture")

pdf.question("Describe the overall architecture.")
pdf.answer(
    "It is a PatchTST-style channel-independent transformer. The raw time series is split into "
    "patches, embedded, passed through a transformer encoder, then augmented with frequency and "
    "dataset embeddings before going through a projector MLP. Each dataset has its own task-specific "
    "head -- either a linear layer for classification or a small MLP for RUL regression. The entire "
    "backbone is shared across all four datasets."
)

pdf.question("Why a transformer and not an LSTM or CNN?")
pdf.answer(
    "Transformers capture long-range dependencies through self-attention, which is important for "
    "vibration signals where fault patterns may be distributed across the entire window. They also "
    "parallelize better than LSTMs. Compared to CNNs, the patch-based approach gives more flexibility "
    "-- you can control the receptive field by adjusting patch size instead of stacking convolutional layers."
)

pdf.question("What is PatchTST?")
pdf.answer(
    "PatchTST is an architecture from Nie et al. (2023) designed for time-series forecasting. The key ideas "
    "are: split the signal into patches instead of processing point-by-point, process each channel "
    "independently through the same backbone, and use a transformer encoder. It was adapted for PHM by "
    "adding domain-aware embeddings and multi-task heads."
)

# Data flow summary
pdf.ln(3)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Data Flow Summary:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.set_font("Courier", "", 9)
pdf.set_text_color(30, 30, 30)
flow = (
    "Input (B, C, 2560)\n"
    "  --> Channel-independent reshape: (B*C, 2560)\n"
    "  --> Patch Embedding: (B*C, 79, 128)\n"
    "  --> + Learnable Positional Encoding\n"
    "  --> Transformer Encoder (4 layers): (B*C, 79, 128)\n"
    "  --> Mean pool over patches: (B*C, 128)\n"
    "  --> Reshape + mean over channels: (B, 128)\n"
    "  --> Concat [backbone | freq_embed | dataset_embed]: (B, 192)\n"
    "  --> Projector MLP: (B, 128)\n"
    "  --> Task Head: classification logits or RUL scalar"
)
pdf.multi_cell(0, 4.5, flow, new_x="LMARGIN", new_y="NEXT")

# ── 2. PATCH EMBEDDING ──
pdf.add_page()
pdf.section_title("2. Patch Embedding")

pdf.question("How does the patch embedding work?")
pdf.answer(
    "The raw 1D signal of length 2560 is unfolded into overlapping patches using torch.unfold. Each "
    "patch is 64 samples long with a stride of 32 -- so 50% overlap. Each patch is then linearly "
    "projected from dimension 64 to the model dimension of 128. This gives a sequence of 79 token "
    "vectors, each representing a local segment of the signal."
)

pdf.question("Why 64-sample patches?")
pdf.answer(
    "At 25.6 kHz, 64 samples = 2.5 ms, which is enough to capture local vibratory patterns like "
    "ball-pass frequencies in bearings. The overlap ensures features at patch boundaries are not missed. "
    "An ablation with patch sizes 32, 64, and 128 showed 64 was the best trade-off between resolution "
    "and sequence length."
)

pdf.question("How many patches per window?")
pdf.answer(
    "For a window of 2560 samples with patch_size=64 and patch_stride=32: "
    "(2560 - 64) / 32 + 1 = 79 patches. The transformer sees a sequence of 79 tokens."
)

# ── 3. POSITIONAL ENCODING ──
pdf.section_title("3. Positional Encoding")

pdf.question("What type of positional encoding?")
pdf.answer(
    "Learnable positional encoding -- not sinusoidal. It is a parameter matrix of shape "
    "(1, max_patches, d_model) initialized with small random values (std=0.02), added to the patch "
    "embeddings. Learnable was chosen over sinusoidal because the patch positions have a fixed, bounded "
    "range, so there is no need for the generalization of sinusoidal encoding to unseen lengths."
)

# ── 4. TRANSFORMER ENCODER ──
pdf.section_title("4. Transformer Encoder")

pdf.question("What are the transformer specifications?")
pdf.answer(
    "4 encoder layers, 8 attention heads, d_model=128, feedforward dimension=256, GELU activation, "
    "10% dropout. It uses pre-norm (LayerNorm before attention and FFN, not after) -- norm_first=True "
    "in PyTorch. Pre-norm is more stable during training and has become the standard since GPT-2."
)

pdf.ln(2)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 6, "Transformer Hyperparameters:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
specs = [
    ("Layers", "4"),
    ("Attention heads", "8"),
    ("d_model", "128"),
    ("FFN dimension", "256"),
    ("Activation", "GELU"),
    ("Dropout", "0.1 (10%)"),
    ("Normalization", "Pre-norm (LayerNorm before attention/FFN)"),
    ("Sequence length", "79 patches"),
]
for k, v in specs:
    pdf.key_value(k, v)

pdf.question("Why pre-norm instead of post-norm?")
pdf.answer(
    "Pre-norm places LayerNorm before the self-attention and feedforward blocks rather than after. "
    "This makes gradient flow more stable, especially in deeper models. It converges faster and "
    "requires less hyperparameter tuning."
)

pdf.question("Why only 4 layers?")
pdf.answer(
    "The signal windows are relatively short -- 2560 samples = 0.1 seconds at 25.6 kHz. With 79 "
    "patches, 4 layers is enough to build global context. An ablation sweeping 2, 4, and 6 layers "
    "showed that beyond 4, the model starts overfitting on smaller datasets like CWRU without "
    "meaningful accuracy gain."
)

pdf.question("How does pooling work after the transformer?")
pdf.answer(
    "Mean pooling over the patch dimension. After the transformer, the output is (B, 79, 128) -- "
    "the mean across the 79 patches produces a single (B, 128) vector. This is simpler than a [CLS] "
    "token and works well because all patches contribute to the global fault signature."
)

# ── 5. CHANNEL-INDEPENDENT PROCESSING ──
pdf.add_page()
pdf.section_title("5. Channel-Independent Processing")

pdf.question("How do you handle different numbers of channels across datasets?")
pdf.answer(
    "Channel-independent processing. Each channel is treated as an independent 1D signal and goes "
    "through the exact same backbone separately. For a CMAPSS sample with 14 sensors, (B, 14, 2560) "
    "is reshaped to (B*14, 2560), passed through the transformer to get (B*14, 128), reshaped back "
    "to (B, 14, 128), then averaged across the 14 channels to produce (B, 128).\n\n"
    "For CWRU with 1 channel, there is no averaging -- the single channel output is the representation. "
    "All channels share weights -- this is parameter-efficient and forces the backbone to learn features "
    "that generalize across sensor types."
)

pdf.question("Why average and not concatenate?")
pdf.answer(
    "Concatenation would make the representation dimension dependent on the number of channels, which "
    "varies from 1 to 14 across datasets. Averaging keeps a fixed d_model=128 output regardless of "
    "channel count, which is necessary for the shared projector and heads to work."
)

pdf.question("What about inter-channel relationships? Doesn't averaging lose that?")
pdf.answer(
    "Yes, this is a known trade-off of the channel-independent approach. The PatchTST paper showed "
    "that for many time-series tasks, channel-independent processing actually outperforms channel-mixing "
    "because it acts as a regularizer and prevents overfitting to spurious cross-channel correlations. "
    "For future work, a lightweight cross-channel attention layer could be added after the per-channel "
    "transformer."
)

# ── 6. FREQUENCY & DATASET EMBEDDINGS ──
pdf.section_title("6. Frequency & Dataset Embeddings")

pdf.question("What is the frequency embedding and why is it needed?")
pdf.answer(
    "The four datasets have sampling rates from 1 Hz (CMAPSS, cycle-based) to 97 kHz (MFPT). The "
    "same vibration pattern at 12 kHz versus 97 kHz looks completely different in the time domain. "
    "The frequency embedding gives the model awareness of the time scale.\n\n"
    "It takes log10(frequency) as input -- log-scale because the range spans 5 orders of magnitude "
    "-- and passes it through a small MLP (1 -> 64 -> 32) with ReLU activations to get a 32-dimensional "
    "embedding. This is concatenated with the backbone output before the projector."
)

pdf.question("What is the dataset embedding?")
pdf.answer(
    "A standard learned embedding lookup table -- same idea as token embeddings in NLP. Each of the "
    "4 datasets gets a 32-dimensional learned vector. This allows the model to learn dataset-specific "
    "biases. It is concatenated alongside the frequency embedding, giving the projector a total input "
    "of 128 + 32 + 32 = 192 dimensions."
)

pdf.question("Could the model work without these embeddings?")
pdf.answer(
    "There are flags use_freq_embed and use_dataset_embed to ablate them. Without frequency embedding, "
    "the model confuses signals from different sampling rates. Without dataset embedding, performance "
    "drops slightly because the projector has to implicitly learn domain distinctions."
)

# ── 7. PROJECTOR MLP ──
pdf.add_page()
pdf.section_title("7. Projector MLP")

pdf.question("What does the projector do?")
pdf.answer(
    "It maps from the 192-dimensional concatenated feature (backbone + frequency + dataset embeddings) "
    "down to a 128-dimensional latent space. It is a single linear layer followed by GELU and 10% "
    "dropout. This latent vector is what the task heads operate on -- the shared representation layer "
    "that serves as the common language across all domains."
)

# ── 8. TASK HEADS ──
pdf.section_title("8. Task Heads")

pdf.question("How are classification and regression handled?")
pdf.answer(
    "Each dataset has its own dedicated heads stored in a ModuleDict.\n\n"
    "Classification heads: a single linear layer from 128 to the number of classes -- 4 for CWRU, "
    "2 for CMAPSS, 3 for MFPT.\n\n"
    "RUL regression head: a 3-layer MLP: Linear(128->128) -> GELU -> Dropout(0.1) -> Linear(128->64) "
    "-> GELU -> Linear(64->1) -> Hardtanh(0,1). Hardtanh clamps the output to [0,1] since RUL targets "
    "are min-max normalized."
)

pdf.question("Why separate heads per dataset instead of one shared head?")
pdf.answer(
    "Each dataset has a different label space. CWRU has 4 bearing fault types, MFPT has 3, CMAPSS has 2. "
    "One classification head cannot serve different class sets. For RUL, separate heads let each dataset "
    "learn its own degradation curve mapping -- bearing degradation (PRONOSTIA) has a very different "
    "profile than turbofan degradation (CMAPSS)."
)

pdf.question("Why Hardtanh and not Sigmoid on the regression head?")
pdf.answer(
    "Sigmoid squashes gradients toward zero as the output approaches 0 or 1 -- exactly where RUL "
    "matters most. A bearing at end-of-life has RUL near 0, and a healthy one near 1. Hardtanh clips "
    "to [0,1] but keeps linear gradients within that range, so the model can still learn effectively "
    "at the extremes."
)

# ── 9. DATA PIPELINE ──
pdf.section_title("9. Data Pipeline")

pdf.question("How is the input data preprocessed?")
pdf.answer(
    "Four steps:\n"
    "1. Load raw signals from each dataset via the phmd library.\n"
    "2. Resample all time-domain signals to a common 25.6 kHz using scipy's Fourier-based resampling "
    "-- except CMAPSS which stays cycle-based.\n"
    "3. Window into 2560-sample segments with 50% overlap (stride = 1280).\n"
    "4. Per-channel z-score normalization within each window -- zero mean, unit variance."
)

pdf.question("Why per-channel z-score and not global normalization?")
pdf.answer(
    "Different sensors have wildly different value ranges -- accelerometers in g versus temperature "
    "in Kelvin. Per-channel z-score makes each channel zero-mean and unit-variance within each window, "
    "so the model sees normalized patterns regardless of physical units. Doing it per-window rather "
    "than per-dataset also makes the model robust to non-stationarity."
)

pdf.question("How is RUL normalized?")
pdf.answer(
    "Min-max normalization with a clip at 125 cycles (the standard CMAPSS convention). Raw RUL is "
    "clipped to [0, 125], then divided by 125 to get [0, 1]. This piecewise-linear clamping is "
    "standard because early RUL values (far from failure) are unreliable and uncorrelated with "
    "degradation."
)

# ── 10. TRAINING STRATEGY ──
pdf.add_page()
pdf.section_title("10. Training Strategy")

pdf.question("How is training done across multiple datasets?")
pdf.answer(
    "All datasets are combined into a single HDF5 file. A WeightedRandomSampler balances batches so "
    "smaller datasets are not drowned out -- each sample is weighted inversely to its dataset size. "
    "The multi-task loss is: cls_weight * CrossEntropy + rul_weight * MSE, with masking: samples "
    "without a classification label (marked with sentinel -1) skip the CE term, and those without "
    "RUL targets skip the MSE term."
)

pdf.question("What optimizer and schedule?")
pdf.answer(
    "AdamW with lr=0.0003 and weight_decay=0.01. The schedule is: 5 epochs of linear warmup from 1% "
    "of the target LR, then cosine annealing for the remaining 55 epochs. Gradient clipping at norm "
    "1.0. Mixed precision training (AMP) on GPU. Early stopping with patience 15 based on a combined "
    "metric of accuracy + (1 - MAE)."
)

pdf.ln(2)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 6, "Training Hyperparameters:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
train_specs = [
    ("Optimizer", "AdamW"),
    ("Learning rate", "0.0003"),
    ("Weight decay", "0.01"),
    ("Batch size", "128"),
    ("Epochs", "60"),
    ("Warmup", "5 epochs, linear from 1% LR"),
    ("LR schedule", "Cosine annealing after warmup"),
    ("Gradient clipping", "Max norm 1.0"),
    ("Mixed precision", "Yes (AMP on CUDA)"),
    ("Early stopping", "Patience 15, metric = acc + (1 - MAE)"),
    ("cls_loss_weight", "1.0"),
    ("rul_loss_weight", "1.0"),
    ("Sampler", "WeightedRandomSampler (balanced domains)"),
]
for k, v in train_specs:
    pdf.key_value(k, v)

# ── 11. FINE-TUNING ──
pdf.section_title("11. Fine-Tuning Strategy")

pdf.question("How does fine-tuning work?")
pdf.answer(
    "Three-stage transfer learning:\n\n"
    "Stage 1 -- Freeze backbone (10-20 epochs): Freeze the entire transformer backbone, train only "
    "the task heads. LR = 0.001. This calibrates the heads to the pretrained features.\n\n"
    "Stage 2 -- Partial unfreeze (15 epochs): Unfreeze the last 2 transformer layers plus embeddings "
    "and projector. LR = 0.00005. Adapts high-level features.\n\n"
    "Stage 3 -- Full fine-tune (30 epochs): Unfreeze everything. LR = 0.00005. Full end-to-end "
    "fine-tuning.\n\n"
    "Each stage has its own early stopping (patience 10) and cosine LR schedule. The best model "
    "from each stage is loaded before starting the next stage."
)

pdf.question("Why three stages instead of just unfreezing everything?")
pdf.answer(
    "Progressive unfreezing prevents catastrophic forgetting. If everything is unfrozen at once, "
    "the early transformer layers -- which have learned general low-level features -- get overwritten "
    "by a single dataset's gradients. By freezing them initially and only unfreezing later at a lower "
    "LR, the general features are preserved while the model adapts to the target domain."
)

pdf.ln(2)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 6, "Fine-Tuning Hyperparameters:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
ft_specs = [
    ("Stage 1 epochs", "10 (20 for regression-only datasets)"),
    ("Stage 1 LR", "0.001 (heads only)"),
    ("Stage 2 epochs", "15"),
    ("Stage 2 LR", "0.00005 (backbone)"),
    ("Stage 3 epochs", "30"),
    ("Stage 3 LR", "0.00005 (full model)"),
    ("Early stopping", "Patience 10 per stage"),
    ("Classification loss", "CrossEntropyLoss"),
    ("Regression loss", "SmoothL1Loss (Huber, more robust than MSE)"),
]
for k, v in ft_specs:
    pdf.key_value(k, v)

# ── 12. DATASETS ──
pdf.add_page()
pdf.section_title("12. Dataset Specifications")

pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(30, 30, 30)

datasets_info = [
    ("CWRU", "Bearing fault diagnosis", "1 channel (drive-end accelerometer)",
     "12,000 Hz", "Classification -- 4 classes (normal, inner race, ball, outer race)",
     "2560 samples at 25.6 kHz = 0.1s windows"),
    ("PRONOSTIA", "Bearing degradation / run-to-failure", "2 channels (vertical + horizontal accelerometer)",
     "25,600 Hz", "RUL regression",
     "2560 samples at 25.6 kHz = 0.1s windows"),
    ("CMAPSS", "Turbofan engine degradation (NASA)", "14 sensor channels (T2, T24, T30, T50, P2, P15, P30, Nf, Nc, Ps30, phi, NRf, NRc, BPR)",
     "1 Hz (cycle-based)", "Classification (2 classes) + RUL regression",
     "50 cycles per window (not resampled)"),
    ("MFPT", "Bearing fault diagnosis", "1 channel (vibration)",
     "97,656 Hz", "Classification -- 3 classes (normal, inner race, outer race)",
     "2560 samples at 25.6 kHz = 0.1s windows"),
]

for name, domain, channels, freq, task, windowing in datasets_info:
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(0, 8, name, new_x="LMARGIN", new_y="NEXT")
    pdf.key_value("Domain", domain)
    pdf.key_value("Channels", channels)
    pdf.key_value("Sampling rate", freq)
    pdf.key_value("Task", task)
    pdf.key_value("Windowing", windowing)
    pdf.ln(3)

# ── 13. MODEL SIZE ──
pdf.section_title("13. Model Size")

pdf.question("How many parameters?")
pdf.answer(
    "Under 1 million total. Approximate breakdown:\n"
    "- Backbone (transformer + patch embed + positional encoding + LayerNorm): ~530K\n"
    "- Task heads (all classification + regression heads): ~50K\n"
    "- Embeddings (frequency + dataset): ~5K\n"
    "- Projector MLP: ~25K\n\n"
    "It is deliberately lightweight -- this is a proof of concept that cross-domain pretraining "
    "helps even with a small model. Scaling up is a natural next step."
)

# ── 14. RESULTS SUMMARY ──
pdf.add_page()
pdf.section_title("14. Results Summary")

pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Classification Performance:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

cls_results = [
    ("CWRU", "Baseline: 92.75%", "Foundation: 97.95%", "Gain: +5.20%"),
    ("MFPT", "Baseline: 98.29%", "Foundation: 98.29%", "Gain: +0.00%"),
    ("Average", "Baseline: 95.52%", "Foundation: 98.12%", "Gain: +2.60%"),
]
for name, bl, fd, gain in cls_results:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(25, 5.5, name)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 5.5, bl)
    pdf.cell(50, 5.5, fd)
    pdf.set_text_color(0, 120, 60)
    pdf.cell(0, 5.5, gain, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(30, 30, 30)

pdf.ln(3)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "F1 Scores (Macro):", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

f1_results = [
    ("CWRU", "Baseline: 93.96%", "Foundation: 98.27%", "Gain: +4.31%"),
    ("MFPT", "Baseline: 97.53%", "Foundation: 97.53%", "Gain: +0.00%"),
]
for name, bl, fd, gain in f1_results:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(25, 5.5, name)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 5.5, bl)
    pdf.cell(50, 5.5, fd)
    pdf.set_text_color(0, 120, 60)
    pdf.cell(0, 5.5, gain, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(30, 30, 30)

pdf.ln(3)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Cross-Domain Generalization (Leave-One-Out):", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(30, 30, 30)
pdf.key_value("CWRU held out", "Accuracy: 96.38%  |  F1: 96.91%  (trained on PRONOSTIA + CMAPSS + MFPT)")
pdf.key_value("MFPT held out", "Accuracy: 97.44%  |  F1: 96.34%  (trained on CWRU + PRONOSTIA + CMAPSS)")

pdf.ln(3)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Low-Data Regime (Baseline CNN on CWRU):", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)

pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(30, 30, 30)
low_data = [("10% data", "42.43%"), ("20% data", "41.38%"), ("50% data", "48.33%"), ("100% data", "48.60%")]
for frac, acc in low_data:
    pdf.key_value(frac, acc)


# ── Save ──
out = "results/architecture_reference.pdf"
pdf.output(out)
print(f"PDF saved to {out}")
