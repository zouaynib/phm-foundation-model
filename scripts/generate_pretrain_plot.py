"""Generate pre-training loss curve plot from pretrain_log.csv"""
import matplotlib.pyplot as plt
import csv
import os

# Read pretrain log
epochs, train_losses, val_losses = [], [], []
csv_path = os.path.join(os.path.dirname(__file__), '..', 'results_files', 'pretrain_log.csv')
if not os.path.exists(csv_path):
    csv_path = 'pretrain_log.csv'  # fallback

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        train_losses.append(float(row['train_loss']))
        val_losses.append(float(row['val_loss']))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_losses, label='Train Loss', color='#2196F3', linewidth=2)
ax.plot(epochs, val_losses, label='Validation Loss', color='#FF5722', linewidth=2)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Loss (MSE, summed over patch dim)', fontsize=13)
ax.set_title('Pre-Training Convergence (MAE)', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 200)

# Annotate the sharp drop
ax.annotate('Sharp drop\n(cross-domain\nstructure found)',
            xy=(40, 45), fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'),
            xytext=(70, 52))

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'plots', 'pretrain_loss.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved to {out_path}")
