"""
HW8 DeepSVDD Baseline — Deep One-Class Classification
Paper: Ruff et al., "Deep One-Class Classification" (ICML 2018)
       http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf

Core idea (FUNDAMENTALLY DIFFERENT from autoencoders):
  - NO reconstruction. NO decoder.
  - Train an encoder phi(x; W) that maps inputs into R^d
  - Goal: shrink all NORMAL samples toward a single fixed center c
  - Loss = mean(||phi(x) - c||^2)
  - At inference: anomaly_score = ||phi(x) - c||^2
                  (anomalies should be FAR from c, normals close)

Critical implementation details (to prevent the trivial all-zeros solution):
  1. NO bias terms in any conv/linear layer
  2. BatchNorm with affine=False (no learnable shift)
  3. Center c is FIXED after init -- computed as the mean of initial features,
     then frozen forever. If c was learnable, model could just move c to the
     features. If c=0, model could just zero out all features.
  4. Eps trick on center: any dim of c too close to 0 gets pushed to +/-eps
     (otherwise the model would push features toward zero dims)
  5. Weight decay (small): regularize encoder weights

Hypothesis: this brings genuinely different anomaly signal because it has
            NO reconstruction. Expected corr with our AE family << 0.99.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'deepsvdd_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'deepsvdd_model.pt')
CENTER_PATH = os.path.join(OUTPUT_DIR, 'deepsvdd_center.pt')

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-4              # smaller than usual for stability
WEIGHT_DECAY = 5e-7    # paper value
FEATURE_DIM = 128      # paper used 128 for CIFAR
CENTER_EPS = 0.1       # any |c_i| < eps gets pushed to +/- eps
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FaceDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1)


class DeepSVDDEncoder(nn.Module):
    """Conv encoder that maps 64x64x3 to a feature vector in R^FEATURE_DIM.
    No bias terms anywhere (critical to prevent trivial all-zeros solution).
    """
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        # bias=False everywhere; BatchNorm with affine=False
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4, feature_dim, bias=False)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc(h)


@torch.no_grad()
def init_center(model, loader, eps=CENTER_EPS):
    """Compute the initial center c as the mean of features over all training data.
    Then apply eps trick: any dim where |c_i| < eps gets pushed to +/-eps.
    The center is then FROZEN -- never trained.
    """
    model.eval()
    n = 0
    c = torch.zeros(FEATURE_DIM, device=device)
    for x in tqdm(loader, desc='Init center', leave=False):
        x = x.to(device, non_blocking=True)
        out = model(x)
        c += out.sum(dim=0)
        n += out.size(0)
    c /= n
    # Eps trick: push small |c_i| away from 0
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c


def train_one_epoch(model, loader, optimizer, c):
    model.train()
    total = 0.0
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        z = model(x)
        # Loss = mean squared distance from frozen center
        loss = ((z - c) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def compute_anomaly_scores(model, loader, c):
    model.eval()
    scores = []
    for x in tqdm(loader, desc='Inference', leave=False):
        x = x.to(device, non_blocking=True)
        z = model(x)
        d2 = ((z - c) ** 2).sum(dim=1)
        scores.extend(d2.cpu().tolist())
    return scores


def main():
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_loader = DataLoader(
        FaceDataset(os.path.join(DATA_DIR, 'trainingset.npy')),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        FaceDataset(os.path.join(DATA_DIR, 'testingset.npy')),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    # For center init: use a non-shuffled loader of training data
    init_loader = DataLoader(
        FaceDataset(os.path.join(DATA_DIR, 'trainingset.npy')),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = DeepSVDDEncoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}  (feature_dim={FEATURE_DIM})')

    print('\n=== Computing center c from initial forward pass ===')
    c = init_center(model, init_loader)
    print(f'Center stats: mean={c.mean().item():.4f}, std={c.std().item():.4f}, '
          f'min={c.min().item():.4f}, max={c.max().item():.4f}')
    print(f'  (norm: {c.norm().item():.4f})')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print('\n=== Training ===')
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, c)
        scheduler.step()
        # Watch for degenerate solution: if loss collapses to near 0, model is
        # trivially mapping everything to c (or zeroing out everything).
        feat_norm = None
        if epoch % 10 == 0 or epoch in (1, 2, 3):
            with torch.no_grad():
                model.eval()
                # Sample-based feature norm to detect degeneracy
                sample_x = next(iter(train_loader)).to(device)
                sample_z = model(sample_x)
                feat_norm = sample_z.norm(dim=1).mean().item()
        suffix = f' | feat_norm {feat_norm:.3f}' if feat_norm is not None else ''
        print(f'Epoch {epoch:3d}/{EPOCHS} | loss {avg_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}{suffix}')

    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(c, CENTER_PATH)
    print(f'Saved model -> {MODEL_PATH}')
    print(f'Saved center -> {CENTER_PATH}')

    print('\n=== Inference ===')
    scores = compute_anomaly_scores(model, test_loader, c)
    print(f'Score stats: min={min(scores):.4f}, max={max(scores):.4f}, '
          f'mean={np.mean(scores):.4f}, std={np.std(scores):.4f}')

    with open(OUTPUT_CSV, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    print(f'Wrote {len(scores)} scores -> {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
