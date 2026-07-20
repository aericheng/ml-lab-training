"""
Ensemble with CutPaste Mahalanobis added to existing best ensemble.

CutPaste mahal has 0.725 AUC alone (close to medium 0.729) AND low
correlations with everything else (0.28-0.39). This is the strongest
ensemble candidate we've found.

Combinations to try:
  v1: 3seed + vae + cutpaste 0.3x  (light)
  v2: 3seed + vae + cutpaste 0.5x  (moderate)
  v3: 3seed + vae + cutpaste 1.0x  (full weight)
  v4: 3seed + vae + ds 0.3x + cutpaste 0.3x  (5-way, current best + cutpaste)
  v5: memae + cutpaste only (extreme test of complementarity)
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
    s42   = load('memae_v1')
    s123  = load('memae_v1_seed123')
    s2024 = load('memae_v1_seed2024')
    vae   = load('vae')
    ds    = load('deepsvdd')
    cp    = load('cutpaste_mahal')

    variants = {
        # === First round (already submitted) ===
        '3seed_vae_cp03':         [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 0.3)],
        '3seed_vae_cp05':         [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 0.5)],
        '3seed_vae_cp1':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 1.0)],

        # === New: heavier CutPaste weights (cp1 won the first round) ===
        '3seed_vae_cp15':         [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 1.5)],
        '3seed_vae_cp2':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 2.0)],
        '3seed_vae_cp3':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3.0)],
        '3seed_vae_cp5':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 5.0)],

        # === Drop VAE, push CutPaste heavier ===
        '3seed_cp1':              [(s42, 1), (s123, 1), (s2024, 1), (cp, 1.0)],
        '3seed_cp2':              [(s42, 1), (s123, 1), (s2024, 1), (cp, 2.0)],
        '3seed_cp3':              [(s42, 1), (s123, 1), (s2024, 1), (cp, 3.0)],

        # === Reduce MemAE weight (let CP dominate) ===
        '3seed05_vae_cp2':        [(s42, 0.5), (s123, 0.5), (s2024, 0.5), (vae, 1), (cp, 2.0)],
        '3seed05_vae_cp3':        [(s42, 0.5), (s123, 0.5), (s2024, 0.5), (vae, 1), (cp, 3.0)],

        # === Build on cp winning recipe + DS ===
        '3seed_vae_cp1_ds03':     [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 1.0), (ds, 0.3)],
        '3seed_vae_cp2_ds03':     [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 2.0), (ds, 0.3)],

        # === Round 3: probe the saturation curve ===
        '3seed_vae_cp4':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 4.0)],
        '3seed_vae_cp5':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 5.0)],

        # cp3 was best Public; try variants on it
        '3seed_vae_cp3_ds03':     [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3.0), (ds, 0.3)],
        '3seed_vae_cp3_ds05':     [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3.0), (ds, 0.5)],

        # cp2_ds03 was best Private; try more DS
        '3seed_vae_cp2_ds05':     [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 2.0), (ds, 0.5)],
    }

    print('=== Building CutPaste ensemble variants ===\n')
    for name, parts in variants.items():
        z = zscore_ensemble(parts)
        r = rank_ensemble(parts)
        write(f'ensemble_z_{name}', z)
        write(f'ensemble_r_{name}', r)

        corr_z_cp = np.corrcoef(z, cp)[0, 1]
        corr_z_s42 = np.corrcoef(z, s42)[0, 1]
        print(f'  {name:30s}  z: corr_w_cp={corr_z_cp:+.3f}  corr_w_memae={corr_z_s42:+.3f}')


if __name__ == '__main__':
    main()
