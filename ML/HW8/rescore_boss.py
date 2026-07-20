"""
Re-score using boss model with different score formulas to find the best.

Hypothesis: classifier head is misleading because anime faces don't look
like noisy faces. Pure denoising AE (MSE only) might work better.
"""
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from boss_baseline import BossAE, FaceDataset

DEVICE = torch.device('cuda')
BATCH_SIZE = 256

model = BossAE().to(DEVICE)
model.load_state_dict(torch.load('output/boss_model.pt'))
model.eval()

loader = DataLoader(
    FaceDataset('data/testingset.npy'),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)

mse_scores = []
cls_fake_scores = []
with torch.no_grad():
    for x in tqdm(loader, desc='Inference'):
        x = x.to(DEVICE)
        recon, z, logit = model(x)
        mse = ((recon - x) ** 2).mean(dim=[1, 2, 3])
        cls_fake = 1.0 - torch.sigmoid(logit)
        mse_scores.extend(mse.cpu().tolist())
        cls_fake_scores.extend(cls_fake.cpu().tolist())

mse_arr = np.array(mse_scores)
cls_arr = np.array(cls_fake_scores)

# Standardize for fair combination
mse_z = (mse_arr - mse_arr.mean()) / (mse_arr.std() + 1e-8)
cls_z = (cls_arr - cls_arr.mean()) / (cls_arr.std() + 1e-8)

# Load medium for ensemble comparison
medium = pd.read_csv('output/medium_submission.csv')['score'].values
medium_z = (medium - medium.mean()) / (medium.std() + 1e-8)

variants = {
    'boss_mse_only':      mse_arr,                          # pure denoising AE
    'boss_cls_only':      cls_arr,                          # pure classifier
    'boss_combined':      mse_z + cls_z,                    # original boss (already submitted)
    'boss_mse_z':         mse_z,                            # boss MSE z-score
    'ensemble_med_bossmse':  medium_z + mse_z,              # medium + boss-MSE
    'ensemble_med_bosscomb': medium_z + (mse_z + cls_z),    # medium + boss combined
}

print('=== Score distribution and correlation with medium ===')
for name, s in variants.items():
    corr = np.corrcoef(s, medium)[0, 1]
    print(f'{name:30s}  mean={s.mean():+.3f}  std={s.std():.3f}  corr_w_medium={corr:+.3f}')

# Write candidates as submission CSVs (we won't submit all)
os.makedirs('output', exist_ok=True)
for name in ('boss_mse_only', 'ensemble_med_bossmse'):
    scores = variants[name]
    out = f'output/{name}_submission.csv'
    with open(out, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    print(f'Wrote {out}')
