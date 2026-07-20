"""
Final ensemble experiments combining all 6 paradigms:
  - 3-seed MemAE v1 (memory-based, ~0.75)
  - VAE (KL regularized, ~0.73)
  - DeepSVDD (one-class, ~0.64)
  - CutPaste Mahalanobis (self-supervised, ~0.73)
  - PatchSVDD inverted (patch + position prediction, ~0.76)

The big question: combining all these uncorrelated signals,
can we push past 0.79?
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
    # Load all available individual model scores
    s42   = load('memae_v1')
    s123  = load('memae_v1_seed123')
    s2024 = load('memae_v1_seed2024')
    vae   = load('vae')
    ds    = load('deepsvdd')
    cp    = load('cutpaste_mahal')
    ps    = load('patchsvdd_inv')   # use inverted!

    variants = {
        # === Drop PatchSVDD weight in to existing best (cp3_ds03 was best Public) ===
        'cp3_ds03_ps05':           [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 0.5)],
        'cp3_ds03_ps1':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 1)],
        'cp3_ds03_ps2':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 2)],
        'cp3_ds03_ps3':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 3)],

        # === Without DS, just cp3 + patchsvdd ===
        'cp3_ps1':                 [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ps, 1)],
        'cp3_ps2':                 [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ps, 2)],
        'cp3_ps3':                 [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ps, 3)],

        # === Pure CutPaste + PatchSVDD (most uncorrelated 2-way) ===
        'cp_ps_balanced':          [(cp, 1), (ps, 1)],
        'cp_ps_cp2x':              [(cp, 2), (ps, 1)],

        # === Best Private: cp2_ds03 + PatchSVDD ===
        'cp2_ds03_ps1':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 2), (ds, 0.3), (ps, 1)],
        'cp2_ds03_ps2':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 2), (ds, 0.3), (ps, 2)],

        # === Round 2: push PS weight higher (ps3 still climbing!) ===
        'cp3_ds03_ps4':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 4)],
        'cp3_ds03_ps5':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 5)],
        'cp3_ds03_ps7':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 7)],
        'cp3_ds03_ps10':           [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 10)],

        # cp_ps_balanced was very strong at 0.819 — push it more
        'cp2_ps2':                 [(cp, 2), (ps, 2)],
        'cp3_ps3':                 [(cp, 3), (ps, 3)],
        'cp_ps2x':                 [(cp, 1), (ps, 2)],
        'cp2_ps3':                 [(cp, 2), (ps, 3)],

        # CP + PS heavy with memae light
        'cp3_ps3_memae_lite':      [(s42, 0.3), (s123, 0.3), (s2024, 0.3), (cp, 3), (ps, 3)],
        'cp3_ps4_memae_lite':      [(s42, 0.3), (s123, 0.3), (s2024, 0.3), (cp, 3), (ps, 4)],

        # === Round 3: Fine-tune around ps=5 (current peak 0.83929) ===
        'cp3_ds03_ps45':           [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 4.5)],
        'cp3_ds03_ps55':           [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 5.5)],
        'cp3_ds03_ps6':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.3), (ps, 6)],

        # Vary DeepSVDD weight at ps=5
        'cp3_ds01_ps5':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.1), (ps, 5)],
        'cp3_ds05_ps5':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ds, 0.5), (ps, 5)],
        'cp3_no_ds_ps5':           [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ps, 5)],

        # Vary CutPaste weight at ps=5
        'cp2_ds03_ps5':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 2), (ds, 0.3), (ps, 5)],
        'cp4_ds03_ps5':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 4), (ds, 0.3), (ps, 5)],
        'cp5_ds03_ps5':            [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 5), (ds, 0.3), (ps, 5)],

        # No VAE
        'cp3_ds03_ps5_no_vae':     [(s42, 1), (s123, 1), (s2024, 1), (cp, 3), (ds, 0.3), (ps, 5)],

        # === Round 4: drop DS pattern proves true, explore further ===
        # No DS exploration around ps=5 (best Public so far 0.83973)
        'cp3_no_ds_ps55':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ps, 5.5)],
        'cp3_no_ds_ps45':          [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ps, 4.5)],
        'cp3_no_ds_ps6':           [(s42, 1), (s123, 1), (s2024, 1), (vae, 1), (cp, 3), (ps, 6)],

        # Drop DS AND VAE
        'cp3_only_ps5':            [(s42, 1), (s123, 1), (s2024, 1), (cp, 3), (ps, 5)],

        # Drop ALL except MemAE+CP+PS
        'only_cpps':               [(cp, 3), (ps, 5)],
        'only_cpps_balanced':      [(cp, 1), (ps, 1)],

        # Heavier MemAE
        'cp3_no_ds_ps5_memae2x':   [(s42, 2), (s123, 2), (s2024, 2), (vae, 1), (cp, 3), (ps, 5)],
        'cp3_no_ds_ps5_memae05':   [(s42, 0.5), (s123, 0.5), (s2024, 0.5), (vae, 1), (cp, 3), (ps, 5)],
    }

    print('=== Building final ensemble variants ===\n')
    for name, parts in variants.items():
        z = zscore_ensemble(parts)
        r = rank_ensemble(parts)
        write(f'ensemble_z_{name}', z)
        write(f'ensemble_r_{name}', r)

        corr_z_ps = np.corrcoef(z, ps)[0, 1]
        corr_z_cp = np.corrcoef(z, cp)[0, 1]
        corr_z_s42 = np.corrcoef(z, s42)[0, 1]
        print(f'  {name:25s}  z: corr_ps={corr_z_ps:+.3f}  corr_cp={corr_z_cp:+.3f}  corr_memae={corr_z_s42:+.3f}')


if __name__ == '__main__':
    main()
