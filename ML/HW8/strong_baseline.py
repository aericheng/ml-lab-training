"""
HW8 Strong Baseline — Multi-Encoder Autoencoder
Target AUC: ~0.77

What's new vs medium:
  Three parallel encoders with different inductive biases (different kernel
  size / dilation) -> concat their latents -> one shared decoder.

  - Encoder A: standard 4x4 conv stack (same as medium)
  - Encoder B: 3x3 conv stack (smaller kernels, finer features)
  - Encoder C: 4x4 conv with dilation=2 in mid layers (larger effective
              receptive field, catches macro face structure)

  Each encoder learns a different "view" of the face distribution.
  Concatenated latent -> the decoder must use all three views to reconstruct.
  Anomalies (anime / cartoon) tend to look weird from at least one view,
  so the combined reconstruction error is sharper.

  Ref: Multi-encoder anomaly detection variants, e.g.
       https://arxiv.org/abs/2003.04060
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'strong_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'strong_model.pt')

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 256   # per-encoder latent; concat -> 3 * 256 = 768
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


def conv_block(in_c, out_c, k=4, s=2, p=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


class EncoderA(nn.Module):
    """Standard 4x4 stride-2 conv. 64 -> 4."""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc(h.flatten(1))


class EncoderB(nn.Module):
    """3x3 stride-2 convs. Finer-grained spatial features."""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, 32, k=3, s=2, p=1),
            conv_block(32, 64, k=3, s=2, p=1),
            conv_block(64, 128, k=3, s=2, p=1),
            conv_block(128, 256, k=3, s=2, p=1),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc(h.flatten(1))


class EncoderC(nn.Module):
    """Dilated convs in middle layers for larger effective receptive field."""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, 32),                      # 64 -> 32
            conv_block(32, 64, k=3, s=2, p=2, d=2), # 32 -> 16 (dilated)
            conv_block(64, 128, k=3, s=2, p=2, d=2),# 16 -> 8 (dilated)
            conv_block(128, 256),                   # 8 -> 4
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc(h.flatten(1))


class MultiEncoderAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc_a = EncoderA(latent_dim)
        self.enc_b = EncoderB(latent_dim)
        self.enc_c = EncoderC(latent_dim)
        # Concat 3 latents -> project back to 4x4x256 for shared decoder
        self.fc_dec = nn.Linear(3 * latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, x):
        za = self.enc_a(x)
        zb = self.enc_b(x)
        zc = self.enc_c(x)
        z = torch.cat([za, zb, zc], dim=1)
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder(h)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        recon = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def compute_anomaly_scores(model, loader):
    model.eval()
    scores = []
    for x in tqdm(loader, desc='Inference', leave=False):
        x = x.to(device, non_blocking=True)
        recon = model(x)
        err = ((recon - x) ** 2).mean(dim=[1, 2, 3])
        scores.extend(err.cpu().tolist())
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

    model = MultiEncoderAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print('\n=== Training ===')
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        marker = ' *' if avg_loss < best_loss else ''
        best_loss = min(best_loss, avg_loss)
        print(f'Epoch {epoch:3d}/{EPOCHS} | loss {avg_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}{marker}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model -> {MODEL_PATH}')

    print('\n=== Inference ===')
    scores = compute_anomaly_scores(model, test_loader)

    with open(OUTPUT_CSV, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    print(f'Wrote {len(scores)} scores -> {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
