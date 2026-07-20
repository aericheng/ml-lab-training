"""
Ensemble experiments adding DeepSVDD.

DeepSVDD alone is much weaker (0.637) than MemAE (0.752), but correlates
much less with the rest (corr 0.32 with conv-AE family, 0.49 with MemAE).

Question: can a low-AUC but low-corr model help via ensemble?

We try several DeepSVDD weights. Too high -> drag down the ensemble.
Too low -> no effect.
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
    # All available individual model scores
    s42   = load('memae_v1')
    s123  = load('memae_v1_seed123')
    s2024 = load('memae_v1_seed2024')
    vae   = load('vae')
    ds    = load('deepsvdd')

    # The current best ensemble used: 3seed_vae_balanced (each 1x)
    # We add DeepSVDD with different weights
    variants = {
        # Variants with DeepSVDD
        '3seed_vae_ds01':  [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (ds, 0.1)],
        '3seed_vae_ds03':  [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (ds, 0.3)],
        '3seed_vae_ds05':  [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (ds, 0.5)],
        '3seed_vae_ds1':   [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (ds, 1.0)],
        # MemAE + DeepSVDD direct (most uncorrelated combo)
        'memae_ds_balanced': [(s42, 1), (ds, 1)],
        'memae_ds05':        [(s42, 1), (ds, 0.5)],
        # Full 5-way with DeepSVDD heavier
        '3seed_vae_ds_heavy': [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (ds, 2.0)],
    }

    print('=== Building DeepSVDD ensemble variants ===\n')
    for name, parts in variants.items():
        z = zscore_ensemble(parts)
        r = rank_ensemble(parts)
        write(f'ensemble_z_{name}', z)
        write(f'ensemble_r_{name}', r)

        corr_z_ds = np.corrcoef(z, ds)[0, 1]
        corr_z_s42 = np.corrcoef(z, s42)[0, 1]
        print(f'  {name:25s}  z: corr_w_ds={corr_z_ds:+.3f}  corr_w_memae={corr_z_s42:+.3f}')


if __name__ == '__main__':
    main()
