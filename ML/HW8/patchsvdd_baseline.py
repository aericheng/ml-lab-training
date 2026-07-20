"""
HW8 PatchSVDD Baseline — Patch-level DeepSVDD + Self-supervised position prediction
Paper: Yi & Yoon, "Patch SVDD: Patch-level SVDD for Anomaly Detection and Localization"
       (ACCV 2020) https://arxiv.org/abs/2006.16067

Core idea (combines two losses):
  1. Patch-level DeepSVDD:
     - Encode each patch -> feature vector
     - Train features to lie near a fixed center c (like DeepSVDD but per patch)

  2. Self-supervised position prediction:
     - Sample anchor patch + one of 8 surrounding neighbors
     - Predict the relative position (8-way classification)
     - Forces encoder to learn STRUCTURAL face features
       (which patch goes where in a face)

For 64x64 image:
  - 9-patch 3x3 grid, stride 16, patches are 32x32 (overlapping by 16 px)
  - Anchor = center patch at (16:48, 16:48)
  - Neighbors = 8 surrounding positions

Anomaly inference:
  - For each test image: extract features for all 9 grid patches
  - score = mean (||f_i - c||^2) over patches
  - Anomaly faces (anime) -> some patches deviate strongly from center -> high score
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'patchsvdd_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'patchsvdd_model.pt')
CENTER_PATH = os.path.join(OUTPUT_DIR, 'patchsvdd_center.pt')

BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-4              # smaller like DeepSVDD for stability
WEIGHT_DECAY = 5e-7
FEATURE_DIM = 128
PATCH_SIZE = 32        # 32x32 patches from 64x64 image
SVDD_WEIGHT = 1.0      # weight on DeepSVDD loss vs position prediction
CENTER_EPS = 0.1
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 8 neighbor positions (top-left of each patch) in 3x3 grid (anchor = center)
# Grid positions: (0,0), (0,16), (0,32), (16,0), [16,16=anchor], (16,32), (32,0), (32,16), (32,32)
# We label the 8 non-anchor positions:
NEIGHBOR_POSITIONS = [
    (0, 0),    # 0: top-left
    (0, 16),   # 1: top-center
    (0, 32),   # 2: top-right
    (16, 0),   # 3: middle-left
    (16, 32),  # 4: middle-right
    (32, 0),   # 5: bottom-left
    (32, 16),  # 6: bottom-center
    (32, 32),  # 7: bottom-right
]
ANCHOR_POS = (16, 16)


class PatchSVDDDataset(Dataset):
    """Each sample: (anchor_patch, neighbor_patch, neighbor_position_label)."""
    def __init__(self, npy_path):
        self.data = np.load(npy_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        img = torch.from_numpy(x).permute(2, 0, 1)  # (3, 64, 64)

        # Anchor = center patch
        ay, ax = ANCHOR_POS
        anchor = img[:, ay:ay + PATCH_SIZE, ax:ax + PATCH_SIZE]

        # Random neighbor
        label = np.random.randint(8)
        ny, nx = NEIGHBOR_POSITIONS[label]
        neighbor = img[:, ny:ny + PATCH_SIZE, nx:nx + PATCH_SIZE]

        return anchor, neighbor, label


class InferenceDataset(Dataset):
    """Just yields the full image (so we can extract all 9 patches at inference)."""
    def __init__(self, npy_path):
        self.data = np.load(npy_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1)


class PatchEncoder(nn.Module):
    """Encoder for 32x32 patch -> feature_dim vector. DeepSVDD-style: no bias."""
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
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
        )  # 32x32 -> 4x4
        self.fc = nn.Linear(128 * 4 * 4, feature_dim, bias=False)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc(h)


class PatchSVDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PatchEncoder()
        # Position predictor: takes [f_anchor, f_neighbor] -> 8-way
        self.pos_head = nn.Linear(FEATURE_DIM * 2, 8, bias=False)

    def forward(self, anchor, neighbor):
        f_a = self.encoder(anchor)
        f_n = self.encoder(neighbor)
        logits = self.pos_head(torch.cat([f_a, f_n], dim=-1))
        return f_a, f_n, logits


@torch.no_grad()
def init_center(model, loader, eps=CENTER_EPS):
    """Compute initial center c from anchor features (then freeze)."""
    model.eval()
    n = 0
    c = torch.zeros(FEATURE_DIM, device=device)
    for anchor, _, _ in tqdm(loader, desc='Init center', leave=False):
        anchor = anchor.to(device, non_blocking=True)
        f = model.encoder(anchor)
        c += f.sum(dim=0)
        n += anchor.size(0)
    c /= n
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c


def get_lr(epoch, total, warmup=3):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train_one_epoch(model, loader, optimizer, c, lr_factor):
    model.train()
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lr_factor

    total_pos = 0.0
    total_svdd = 0.0
    total_correct = 0
    total_count = 0
    for anchor, neighbor, label in tqdm(loader, leave=False):
        anchor = anchor.to(device, non_blocking=True)
        neighbor = neighbor.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        f_a, f_n, logits = model(anchor, neighbor)
        pos_loss = F.cross_entropy(logits, label)
        # DeepSVDD-style: features near fixed center c
        svdd_loss = ((f_a - c) ** 2).sum(dim=-1).mean() + \
                    ((f_n - c) ** 2).sum(dim=-1).mean()

        loss = pos_loss + SVDD_WEIGHT * svdd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_pos += pos_loss.item()
        total_svdd += svdd_loss.item()
        total_correct += (logits.argmax(dim=-1) == label).sum().item()
        total_count += anchor.size(0)
    n = len(loader)
    return total_pos / n, total_svdd / n, total_correct / total_count


@torch.no_grad()
def compute_anomaly_scores(model, loader, c):
    """For each test image: extract 9 patches, get distance to c per patch, average."""
    model.eval()
    # All 9 grid patches (anchor + 8 neighbors)
    all_positions = [(16, 16)] + NEIGHBOR_POSITIONS
    scores = []
    for img in tqdm(loader, desc='Inference', leave=False):
        img = img.to(device, non_blocking=True)  # (B, 3, 64, 64)
        B = img.size(0)
        # Stack all 9 patches per image
        all_dist = torch.zeros(B, device=device)
        for (py, px) in all_positions:
            patch = img[:, :, py:py + PATCH_SIZE, px:px + PATCH_SIZE]
            f = model.encoder(patch)
            d2 = ((f - c) ** 2).sum(dim=-1)
            all_dist += d2
        all_dist /= len(all_positions)
        scores.extend(all_dist.cpu().tolist())
    return scores


def main():
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_loader = DataLoader(
        PatchSVDDDataset(os.path.join(DATA_DIR, 'trainingset.npy')),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        InferenceDataset(os.path.join(DATA_DIR, 'testingset.npy')),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = PatchSVDD().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    print('\n=== Computing center c from anchor patches ===')
    c = init_center(model, train_loader)
    print(f'Center stats: norm={c.norm().item():.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print('\n=== Training (position prediction + DeepSVDD) ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS)
        pos_loss, svdd_loss, acc = train_one_epoch(model, train_loader, optimizer, c, lr_factor)
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | pos_loss {pos_loss:.4f} | svdd {svdd_loss:.4f} | '
              f'pos_acc {acc:.4f} | lr {LR * lr_factor:.2e}')

    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(c, CENTER_PATH)
    print(f'Saved model -> {MODEL_PATH}')

    print('\n=== Inference ===')
    scores = compute_anomaly_scores(model, test_loader, c)
    print(f'Score stats: min={min(scores):.4f}, max={max(scores):.4f}, mean={np.mean(scores):.4f}')

    with open(OUTPUT_CSV, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    print(f'Wrote {len(scores)} -> {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
