"""
HW8 Multi-Hypothesis Autoencoder (MH-AE) Baseline
Inspired by [2107.08790] "Anomaly Detection Based on Multiple-Hypothesis
Autoencoder" — multiple decoders that each propose a reconstruction.

Architecture:
  - 1 shared encoder
  - K=3 parallel decoders with DIFFERENT random init
  - Each decoder produces its own reconstruction from the same latent
  - Training: sum the K MSE losses (each decoder learns independently)
  - Inference: anomaly score = MIN over K decoders' MSE
                (intuition: anomalies should be poorly reconstructed by ALL
                 decoders; min is the "best case" — if even the best decoder
                 can't reconstruct it, it's an anomaly)

This is the "soft" variant of MH-AE (no winner-takes-all routing). For
this homework's tight timeline, this is the cleanest implementation. The
key insight is testing whether anomalies fail in MULTIPLE decoder modes.
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'mhae_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'mhae_model.pt')

BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
LATENT_DIM = 256
K_DECODERS = 3              # number of parallel decoders
WARMUP_EPOCHS = 3
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


def make_decoder():
    """One decoder module. Will be instantiated K times with different random init."""
    return nn.Sequential(
        nn.Linear(LATENT_DIM, 256 * 4 * 4),
        nn.Unflatten(1, (256, 4, 4)),
        nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
    )


class MultiHypothesisAE(nn.Module):
    def __init__(self, k=K_DECODERS):
        super().__init__()
        self.k = k
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 4 * 4, LATENT_DIM)
        # K decoders, each gets its own random init (PyTorch default init varies per seed call)
        self.decoders = nn.ModuleList([make_decoder() for _ in range(k)])

    def forward(self, x):
        h = self.encoder_conv(x).flatten(1)
        z = self.fc_enc(h)
        recons = [dec(z) for dec in self.decoders]    # list of K (B, 3, 64, 64)
        return torch.stack(recons, dim=0)             # (K, B, 3, 64, 64)


def get_lr(epoch, total, warmup):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train_one_epoch(model, loader, optimizer, lr_factor):
    model.train()
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lr_factor

    total = 0.0
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        recons = model(x)                                  # (K, B, 3, 64, 64)
        # Sum of K independent MSE losses (each decoder learns)
        loss = 0.0
        for k in range(model.k):
            loss = loss + F.mse_loss(recons[k], x)
        loss = loss / model.k                              # average over decoders
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def compute_anomaly_scores(model, loader, mode='min'):
    """Per-image anomaly score across K decoders' reconstructions.

    mode='min': anomalies should fail ALL decoders -> use min MSE
                (if min is large, even the best decoder couldn't reconstruct)
    mode='mean': average MSE
    """
    model.eval()
    scores_min = []
    scores_mean = []
    for x in tqdm(loader, desc='Inference', leave=False):
        x = x.to(device, non_blocking=True)
        recons = model(x)                                  # (K, B, 3, 64, 64)
        errs = ((recons - x.unsqueeze(0)) ** 2).mean(dim=[2, 3, 4])  # (K, B)
        min_err = errs.min(dim=0).values                   # (B,)
        mean_err = errs.mean(dim=0)                        # (B,)
        scores_min.extend(min_err.cpu().tolist())
        scores_mean.extend(mean_err.cpu().tolist())
    if mode == 'min':
        return scores_min
    return scores_mean


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

    model = MultiHypothesisAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}  (K={K_DECODERS} decoders)')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('\n=== Training ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS, WARMUP_EPOCHS)
        avg_loss = train_one_epoch(model, train_loader, optimizer, lr_factor)
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | loss {avg_loss:.6f} | lr {LR * lr_factor:.2e}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model -> {MODEL_PATH}')

    print('\n=== Inference ===')
    # Save both min and mean variants
    scores_min = compute_anomaly_scores(model, test_loader, mode='min')
    scores_mean = compute_anomaly_scores(model, test_loader, mode='mean')

    csv_min = OUTPUT_CSV.replace('.csv', '_min.csv')
    csv_mean = OUTPUT_CSV.replace('.csv', '_mean.csv')
    for path, scores in [(csv_min, scores_min), (csv_mean, scores_mean)]:
        with open(path, 'w') as f:
            f.write('ID,score\n')
            for i, s in enumerate(scores):
                f.write(f'{i},{s}\n')
        print(f'Wrote {len(scores)} scores -> {path}')


if __name__ == '__main__':
    main()
