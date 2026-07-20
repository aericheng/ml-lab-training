"""
Generate breakthrough visualizations for HW8.

Loads anomaly scores directly from submission CSVs (no model training needed).

Outputs:
  06_score_distributions_all.png  ??8 models' score histograms
  07_correlation_heatmap.png      ??pairwise Pearson correlation
  08_auc_progression.png          ??AUC progression bar chart with baseline thresholds
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HW8_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HW8_DIR, 'output')
IMG_DIR = os.path.join(OUT_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)


submissions = {
    'simple':    'simple_submission.csv',
    'medium':    'medium_submission.csv',
    'vae':       'vae_submission.csv',
    'resnet':    'resnet_submission.csv',
    'memae':     'memae_v1_submission.csv',
    'cutpaste':  'cutpaste_mahal_submission.csv',
    'patchsvdd': 'patchsvdd_inv_submission.csv',
    'deepsvdd':  'deepsvdd_submission.csv',
}

scores = {}
print('Loading submission scores...')
for name, fname in submissions.items():
    path = os.path.join(OUT_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.sort_values('ID').reset_index(drop=True)
        scores[name] = df['score'].values
        print(f'  {name:10s}: range=[{scores[name].min():.4f}, {scores[name].max():.4f}], n={len(scores[name])}')
    else:
        print(f'  ! {name}: not found ({path})')


# ===== IMG 6: Score distributions (8 models) =====
plot_order = ['simple', 'medium', 'vae', 'resnet', 'memae', 'cutpaste', 'patchsvdd', 'deepsvdd']
plot_titles = ['SIMPLE', 'MEDIUM', 'VAE', 'RESNET', 'MEMAE (1st breakthrough)',
               'CUTPASTE Mahal (2nd breakthrough)', 'PATCHSVDD inverted (3rd breakthrough)', 'DEEPSVDD']
colors = ['#888888', '#3674B5', '#3FA34D', '#FFAA00', '#E63946',
          '#16A085', '#E67E22', '#9B59B6']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for ax, name, title, color in zip(axes, plot_order, plot_titles, colors):
    if name not in scores:
        ax.axis('off')
        continue
    s = scores[name]
    ax.hist(s, bins=60, color=color, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_title(f'{title}\nstd={s.std():.4g}, range=[{s.min():.3g}, {s.max():.3g}]',
                 fontsize=11)
    ax.set_xlabel('anomaly score')
    ax.set_ylabel('# test images')
    ax.grid(alpha=0.3)

fig.suptitle('Anomaly Score Distributions — All 8 Component Models', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(IMG_DIR, '06_score_distributions_all.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved 06_score_distributions_all.png')


# ===== IMG 7: Correlation heatmap =====
names = [n for n in plot_order if n in scores]
n_models = len(names)
corr_mat = np.zeros((n_models, n_models))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        corr_mat[i, j] = np.corrcoef(scores[a], scores[b])[0, 1]

fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xticks(range(n_models))
ax.set_yticks(range(n_models))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11, weight='bold')
ax.set_yticklabels(names, fontsize=11, weight='bold')

for i in range(n_models):
    for j in range(n_models):
        v = corr_mat[i, j]
        textcolor = 'white' if abs(v) > 0.5 else 'black'
        weight = 'bold' if i != j else 'normal'
        ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                color=textcolor, fontsize=10, weight=weight)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Pearson correlation', fontsize=11)

ax.set_title(
    'Pairwise Score Correlation — Why Ensemble Works\n'
    'Conv-AE family (corr ~0.99) -> ensemble fails | MemAE/CutPaste (corr ~0.3) -> ensemble wins\n'
    'PatchSVDD (corr ~-0.27) -> "wrong direction" inverted becomes the strongest signal',
    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '07_correlation_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved 07_correlation_heatmap.png')


# ===== IMG 8: AUC progression =====
# (label, Public AUC, color, category)
auc_data = [
    ('Simple',                  0.66943, '#888888', 'baseline'),
    ('Medium',                  0.72877, '#3674B5', 'conv-AE'),
    ('Strong v1 (failed)',      0.70716, '#cc4444', 'failed'),
    ('Boss v1 (failed)',        0.68294, '#cc4444', 'failed'),
    ('VAE',                     0.73048, '#3FA34D', 'conv-AE'),
    ('ResNet AE',               0.72885, '#FFAA00', 'conv-AE'),
    ('Conv-AE Ensemble',        0.73006, '#aaaaaa', 'ensemble (micro-drop)'),
    ('MemAE v1 (1st breakthrough)', 0.75269, '#E63946', '1st breakthrough'),
    ('MemAE v2 (paper-faithful, failed)', 0.67298, '#cc4444', 'failed'),
    ('Ensemble (3seed + VAE)',  0.75759, '#9B59B6', 'ensemble'),
    ('+ DeepSVDD 0.3x',         0.75689, '#9B59B6', 'ensemble'),
    ('+ CutPaste 3x (2nd breakthrough)', 0.77978, '#16A085', '2nd breakthrough'),
    ('+ PatchSVDD 5x (3rd breakthrough)', 0.83973, '#E67E22', '3rd breakthrough'),
]

fig, ax = plt.subplots(figsize=(15, 9))
labels = [d[0] for d in auc_data]
aucs = [d[1] for d in auc_data]
bar_colors = [d[2] for d in auc_data]

bars = ax.barh(labels, aucs, color=bar_colors, edgecolor='black', linewidth=0.5)

# Baseline threshold lines
thresholds = [
    (0.53, 'Simple (0.53)',  '#888888'),
    (0.73, 'Medium (0.73)',  '#3674B5'),
    (0.77, 'Strong (0.77)',  '#FFAA00'),
    (0.80, 'Boss (0.80+)',   '#cc0000'),
]
for thresh, label, color in thresholds:
    ax.axvline(x=thresh, linestyle='--', color=color, alpha=0.7, linewidth=1.8)
    ax.text(thresh, -0.7, label, rotation=0, ha='center', va='top',
            fontsize=10, color=color, weight='bold')

# Annotate AUC values on bars
for bar, auc in zip(bars, aucs):
    width = bar.get_width()
    ax.text(width + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{auc:.4f}', ha='left', va='center', fontsize=10, weight='bold')

ax.set_xlabel('Public AUC', fontsize=12)
ax.set_xlim(0.6, 0.92)
ax.set_title('HW8 Progression — From 0.669 to 0.840 (+0.171)\n'
             'All 4 baselines passed (Simple, Medium, Strong, Boss)',
             fontsize=13)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()  # earliest at top

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '08_auc_progression.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved 08_auc_progression.png')

print('\nAll breakthrough visualizations saved to:', IMG_DIR)

