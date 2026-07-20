"""
Multiple ensemble variants that include MemAE (which has corr 0.73 with the
other models — i.e., it brings genuinely different signal).

We try several combinations to find the best one:
  v1: memae + vae         (2-way, most balanced — VAE is our best non-memory model)
  v2: memae + medium      (2-way, swap VAE for medium)
  v3: memae*2 + vae       (MemAE weighted 2x, since it has different signal)
  v4: memae*2 + vae + resnet  (MemAE 2x + others 1x each)

Each is written as both z-score and rank ensemble.
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


def make_zscore_ensemble(scores_with_weights):
    """scores_with_weights: list of (scores_array, weight) tuples."""
    out = np.zeros_like(next(iter(scores_with_weights))[0])
    total_w = 0.0
    for s, w in scores_with_weights:
        out += w * zscore(s)
        total_w += w
    return out / total_w


def make_rank_ensemble(scores_with_weights):
    out = np.zeros_like(next(iter(scores_with_weights))[0], dtype=np.float64)
    total_w = 0.0
    for s, w in scores_with_weights:
        out += w * rankdata(s)
        total_w += w
    return out / total_w


def write(name, scores):
    path = os.path.join(OUTPUT_DIR, f'{name}.csv')
    with open(path, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    return path


def main():
    # Load all individual model scores
    medium = load('medium')
    vae    = load('vae')
    resnet = load('resnet')
    memae  = load('memae_v1')   # v1: paper-broken but high AUC 0.7527
    memae2 = load('memae_v2')   # v2: paper-faithful with proper sparse attention

    print('=== Building ensemble variants ===\n')

    variants = {
        # Baselines (v1 only, already tried)
        'memae_vae':                  [(memae, 1.0), (vae, 1.0)],
        'memae2x_vae':                [(memae, 2.0), (vae, 1.0)],
        # v2 alone in ensembles
        'memae2_vae':                 [(memae2, 1.0), (vae, 1.0)],
        'memae2_2x_vae':              [(memae2, 2.0), (vae, 1.0)],
        # v1 + v2 combinations (both MemAEs)
        'memae_v1v2':                 [(memae, 1.0), (memae2, 1.0)],
        'memae_v1v2_vae':             [(memae, 1.0), (memae2, 1.0), (vae, 1.0)],
        'memae2x_memae2_2x_vae':      [(memae, 2.0), (memae2, 2.0), (vae, 1.0)],
        'memae_v1v2_vae_resnet':      [(memae, 1.0), (memae2, 1.0), (vae, 1.0), (resnet, 1.0)],
    }

    for name, parts in variants.items():
        z_scores = make_zscore_ensemble(parts)
        r_scores = make_rank_ensemble(parts)
        p1 = write(f'ensemble_z_{name}', z_scores)
        p2 = write(f'ensemble_r_{name}', r_scores)
        # Print correlations with memae (high = MemAE signal preserved)
        corr_z = np.corrcoef(z_scores, memae)[0, 1]
        corr_r = np.corrcoef(r_scores, memae)[0, 1]
        print(f'  {name:30s}  corr_w_memae:  z={corr_z:+.3f}  r={corr_r:+.3f}')

    print(f'\nWrote {2 * len(variants)} ensemble CSVs to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
