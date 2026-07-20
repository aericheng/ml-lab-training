"""
HW8 CutPaste Baseline — Self-Supervised Anomaly Detection
Paper: Li et al., "CutPaste: Self-Supervised Learning for Anomaly Detection
       and Localization" (CVPR 2021)
       https://arxiv.org/abs/2104.04015

Core idea (TOTALLY different paradigm — no reconstruction, no memory):
  Generate synthetic "anomalies" by:
    - CutPaste-Rect: cut a rectangle patch, paste at random location (same image)
    - CutPaste-Scar: cut a thin strip, paste at random location
  Train a 3-way classifier: normal / cutpaste-rect / cutpaste-scar
  At inference:
    1. Classifier-based score: 1 - P(normal | x)
    2. Mahalanobis-based score: distance of test feature from training feature
       distribution (Gaussian fit on training features)

Why this might work for face anomaly detection:
  - Anime faces have "wrong" structural arrangement of features compared to
    real faces — similar in spirit to cut-paste artifacts
  - Classifier learns "what intact normal face looks like" without ever seeing
    anime examples
  - Fundamentally different signal from reconstruction-based methods
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
OUTPUT_CSV_CLS = os.path.join(OUTPUT_DIR, 'cutpaste_cls_submission.csv')
OUTPUT_CSV_MAHAL = os.path.join(OUTPUT_DIR, 'cutpaste_mahal_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'cutpaste_model.pt')

BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
FEATURE_DIM = 256
WARMUP_EPOCHS = 3
SEED = 42

# CutPaste hyperparams (from paper)
AREA_RATIO = (0.02, 0.15)        # rect patch area as fraction of image
ASPECT_RATIO = (0.3, 3.3)         # rect aspect ratio range
SCAR_W_RANGE = (2, 16)            # scar width in pixels
SCAR_H_RANGE = (10, 25)           # scar height in pixels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cutpaste_rect(img):
    """Cut a rectangular patch and paste at random location in same image."""
    _, H, W = img.shape
    area = np.random.uniform(*AREA_RATIO) * H * W
    aspect = np.random.uniform(*ASPECT_RATIO)
    ph = max(2, min(int(np.sqrt(area / aspect)), H - 2))
    pw = max(2, min(int(np.sqrt(area * aspect)), W - 2))
    # Source patch
    sy = np.random.randint(0, H - ph + 1)
    sx = np.random.randint(0, W - pw + 1)
    patch = img[:, sy:sy + ph, sx:sx + pw].clone()
    # Target location
    ty = np.random.randint(0, H - ph + 1)
    tx = np.random.randint(0, W - pw + 1)
    img_aug = img.clone()
    img_aug[:, ty:ty + ph, tx:tx + pw] = patch
    return img_aug


def cutpaste_scar(img):
    """Cut a thin scar and paste at random location."""
    _, H, W = img.shape
    sh = np.random.randint(SCAR_H_RANGE[0], min(SCAR_H_RANGE[1], H // 2))
    sw = np.random.randint(SCAR_W_RANGE[0], min(SCAR_W_RANGE[1], W // 2))
    sy = np.random.randint(0, H - sh + 1)
    sx = np.random.randint(0, W - sw + 1)
    patch = img[:, sy:sy + sh, sx:sx + sw].clone()
    ty = np.random.randint(0, H - sh + 1)
    tx = np.random.randint(0, W - sw + 1)
    img_aug = img.clone()
    img_aug[:, ty:ty + sh, tx:tx + sw] = patch
    return img_aug


class CutPasteDataset(Dataset):
    """Each sample: (image, label) where label is one of:
       0: normal (intact)
       1: cutpaste-rect
       2: cutpaste-scar
    """
    def __init__(self, npy_path):
        self.data = np.load(npy_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).permute(2, 0, 1)

        choice = np.random.randint(3)
        if choice == 0:
            return x, 0
        elif choice == 1:
            return cutpaste_rect(x), 1
        else:
            return cutpaste_scar(x), 2


class PlainDataset(Dataset):
    """For inference: just normal images, no augmentation."""
    def __init__(self, npy_path):
        self.data = np.load(npy_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1)


class CutPasteModel(nn.Module):
    """Conv encoder -> feature -> 3-way classifier."""
    def __init__(self, feature_dim=FEATURE_DIM, num_classes=3):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.feature_head = nn.Linear(256 * 4 * 4, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def features(self, x):
        h = self.encoder_conv(x).flatten(1)
        return self.feature_head(h)

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f), f


def get_lr(epoch, total, warmup):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train_one_epoch(model, loader, optimizer, lr_factor):
    model.train()
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lr_factor

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_count += x.size(0)
    return total_loss / len(loader), total_correct / total_count


@torch.no_grad()
def extract_features(model, loader):
    """Pass all images through encoder, return features (N, feature_dim)."""
    model.eval()
    feats = []
    for x in tqdm(loader, desc='Extract', leave=False):
        x = x.to(device, non_blocking=True)
        _, f = model(x)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()


@torch.no_grad()
def classifier_scores(model, loader):
    """Anomaly score = 1 - P(normal class | x)."""
    model.eval()
    scores = []
    for x in tqdm(loader, desc='Classifier scores', leave=False):
        x = x.to(device, non_blocking=True)
        logits, _ = model(x)
        prob_normal = F.softmax(logits, dim=-1)[:, 0]
        scores.extend((1.0 - prob_normal).cpu().tolist())
    return scores


def mahalanobis_scores(test_feats, train_mean, cov_inv):
    """Mahalanobis distance from test features to training distribution."""
    diff = test_feats - train_mean
    return np.einsum('ni,ij,nj->n', diff, cov_inv, diff)


def fit_gaussian_with_shrinkage(features, shrinkage=0.01):
    """Compute mean and shrinkage-regularized inverse covariance."""
    mean = features.mean(axis=0)
    centered = features - mean
    cov = centered.T @ centered / len(features)
    # Shrinkage: blend with identity for numerical stability
    d = cov.shape[0]
    cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / d * np.eye(d)
    cov_inv = np.linalg.inv(cov)
    return mean, cov_inv


def main():
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_loader = DataLoader(
        CutPasteDataset(os.path.join(DATA_DIR, 'trainingset.npy')),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    # For feature extraction over the whole training set (no augmentation)
    train_plain_loader = DataLoader(
        PlainDataset(os.path.join(DATA_DIR, 'trainingset.npy')),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        PlainDataset(os.path.join(DATA_DIR, 'testingset.npy')),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = CutPasteModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('\n=== Training (3-way classification) ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS, WARMUP_EPOCHS)
        loss, acc = train_one_epoch(model, train_loader, optimizer, lr_factor)
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | loss {loss:.4f} | acc {acc:.4f} | lr {LR * lr_factor:.2e}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model -> {MODEL_PATH}')

    print('\n=== Inference: classifier-based score ===')
    cls_scores = classifier_scores(model, test_loader)
    with open(OUTPUT_CSV_CLS, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(cls_scores):
            f.write(f'{i},{s}\n')
    print(f'Wrote {len(cls_scores)} -> {OUTPUT_CSV_CLS}')

    print('\n=== Inference: Mahalanobis score ===')
    print('  extracting training features...')
    train_feats = extract_features(model, train_plain_loader)
    print(f'  fitting Gaussian on (N={len(train_feats)}, D={train_feats.shape[1]})')
    mean, cov_inv = fit_gaussian_with_shrinkage(train_feats)
    print('  extracting test features...')
    test_feats = extract_features(model, test_loader)
    mahal = mahalanobis_scores(test_feats, mean, cov_inv)
    with open(OUTPUT_CSV_MAHAL, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(mahal):
            f.write(f'{i},{s}\n')
    print(f'Wrote {len(mahal)} -> {OUTPUT_CSV_MAHAL}')

    print('\nMahalanobis stats:')
    print(f'  min={mahal.min():.2f}, max={mahal.max():.2f}, mean={mahal.mean():.2f}')


if __name__ == '__main__':
    main()
