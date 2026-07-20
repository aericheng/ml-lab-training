"""
Ensemble multiple model submissions.

Two ensemble strategies:
  1. Z-score average: standardize each model's scores, then average
  2. Rank average:    rank each model's scores [0, N-1], then average ranks

Rank-based is more robust to outlier scores (one model having a few huge
score wouldn't dominate the result). Z-score is simpler and usually fine
for similarly distributed models.

Edit MODELS dict below to choose which submissions to ensemble.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUTPUT_DIR = 'output'

# Edit this to pick which models to combine
MODELS = {
    'medium': os.path.join(OUTPUT_DIR, 'medium_submission.csv'),
    'vae':    os.path.join(OUTPUT_DIR, 'vae_submission.csv'),
    'resnet': os.path.join(OUTPUT_DIR, 'resnet_submission.csv'),
    'memae':  os.path.join(OUTPUT_DIR, 'memae_submission.csv'),
    # 'simple': os.path.join(OUTPUT_DIR, 'simple_submission.csv'),  # 加進來通常會拉低分數
}


def load_scores(paths):
    dfs = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f'  ⚠️  skip {name}: {p} not found')
            continue
        df = pd.read_csv(p)
        df = df.sort_values('ID').reset_index(drop=True)
        dfs[name] = df['score'].values
    return dfs


def ensemble_zscore(scores_dict, weights=None):
    """Standardize each model, then weighted average."""
    names = list(scores_dict.keys())
    if weights is None:
        weights = {n: 1.0 for n in names}
    out = np.zeros_like(next(iter(scores_dict.values())))
    for name in names:
        s = scores_dict[name]
        z = (s - s.mean()) / (s.std() + 1e-8)
        out += weights[name] * z
    return out / sum(weights.values())


def ensemble_rank(scores_dict, weights=None):
    """Rank each model (higher score = higher rank), then weighted average."""
    names = list(scores_dict.keys())
    if weights is None:
        weights = {n: 1.0 for n in names}
    out = np.zeros_like(next(iter(scores_dict.values())), dtype=np.float64)
    for name in names:
        r = rankdata(scores_dict[name])
        out += weights[name] * r
    return out / sum(weights.values())


def write_csv(path, scores):
    with open(path, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    print(f'  wrote {path}')


def main():
    print(f'Loading {len(MODELS)} model(s)...')
    scores_dict = load_scores(MODELS)
    if len(scores_dict) < 2:
        print('Need at least 2 models for ensemble!')
        return

    # Print pairwise correlations as sanity check
    print(f'\nLoaded models: {list(scores_dict.keys())}')
    names = list(scores_dict.keys())
    print('\nPairwise correlations:')
    print('         ' + '  '.join(f'{n:>8}' for n in names))
    for n1 in names:
        row = [f'{np.corrcoef(scores_dict[n1], scores_dict[n2])[0,1]:+.3f}' for n2 in names]
        print(f'  {n1:>6}  ' + '  '.join(f'{v:>8}' for v in row))

    print('\n=== Writing ensembles ===')

    # Z-score ensemble
    z_scores = ensemble_zscore(scores_dict)
    z_path = os.path.join(OUTPUT_DIR, f'ensemble_zscore_{"_".join(names)}.csv')
    write_csv(z_path, z_scores)

    # Rank ensemble
    r_scores = ensemble_rank(scores_dict)
    r_path = os.path.join(OUTPUT_DIR, f'ensemble_rank_{"_".join(names)}.csv')
    write_csv(r_path, r_scores)

    # Print stats and correlation with best single model (assume first one is medium-tier)
    print('\n=== Ensemble vs individual models ===')
    for ens_name, ens_scores in [('zscore', z_scores), ('rank', r_scores)]:
        print(f'{ens_name}:')
        for name in names:
            corr = np.corrcoef(ens_scores, scores_dict[name])[0, 1]
            print(f'  corr with {name}: {corr:+.4f}')


if __name__ == '__main__':
    main()
