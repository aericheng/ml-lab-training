"""
Multi-seed MemAE v1 ensemble experiments.

3 independently-trained MemAE v1 models (seeds 42, 123, 2024) have
cross-correlations ~0.996. Very similar but not identical.

We try:
  v1: avg of 3 v1 seeds (pure MemAE ensemble)
  v2: avg of 3 v1 seeds + VAE (adds uncorrelated signal)
  v3: avg of 3 v1 seeds (2x) + VAE (MemAE weight higher)
  v4: each seed counts individually + VAE (4-way)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUTPUT_DIR = 'output'


def load(name):
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{name}_submission.csv'))
    return df.sort_values('ID').reset_index(drop=True)['score'].values


def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def zscore_ensemble(parts):
    out = np.zeros_like(parts[0][0])
    total_w = 0.0
    for s, w in parts:
        out += w * zscore(s)
        total_w += w
    return out / total_w


def rank_ensemble(parts):
    out = np.zeros_like(parts[0][0], dtype=np.float64)
    total_w = 0.0
    for s, w in parts:
        out += w * rankdata(s)
        total_w += w
    return out / total_w


def write(name, scores):
    path = os.path.join(OUTPUT_DIR, f'{name}.csv')
    with open(path, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')


def main():
    # Load
    s42   = load('memae_v1')
    s123  = load('memae_v1_seed123')
    s2024 = load('memae_v1_seed2024')
    vae   = load('vae')

    # First compute the simple average of 3 seeds (in z-score space)
    memae_3seed_z = (zscore(s42) + zscore(s123) + zscore(s2024)) / 3.0

    variants = {
        '3seed':                      [(s42, 1.0), (s123, 1.0), (s2024, 1.0)],
        '3seed_vae':                  [(s42, 1.0), (s123, 1.0), (s2024, 1.0), (vae, 3.0)],  # vae weighted 3x to balance
        '3seed_vae_balanced':         [(s42, 1.0), (s123, 1.0), (s2024, 1.0), (vae, 1.0)],  # each counts 1x
        '3seed_2x_vae':               [(s42, 2.0), (s123, 2.0), (s2024, 2.0), (vae, 1.0)],  # MemAEs weighted 2x each
    }

    print('=== Building multi-seed ensemble variants ===\n')
    for name, parts in variants.items():
        z = zscore_ensemble(parts)
        r = rank_ensemble(parts)
        write(f'ensemble_z_{name}', z)
        write(f'ensemble_r_{name}', r)

        # Show correlation vs single best (current best is memae_v1 alone)
        corr_z_vae = np.corrcoef(z, vae)[0, 1]
        corr_z_s42 = np.corrcoef(z, s42)[0, 1]
        print(f'  {name:25s}  z-score: corr_w_vae={corr_z_vae:+.3f}  corr_w_seed42={corr_z_s42:+.3f}')

    # Bonus: also pre-compute and save the "3seed average" as a single CSV
    # in raw score space (mean of 3 raw MSE scores). This is what would happen
    # if you had trained "an effective MemAE that's the average of 3 seeds".
    memae_3seed_raw = (s42 + s123 + s2024) / 3.0
    write('memae_v1_3seed_avg', memae_3seed_raw)
    print(f'\nAlso saved raw 3-seed average: ensemble/memae_v1_3seed_avg.csv')


if __name__ == '__main__':
    main()
