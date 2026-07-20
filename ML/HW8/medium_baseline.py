"""
HW8 Medium Baseline — Deeper Conv AE + BatchNorm + LeakyReLU + LR scheduler
Target AUC: ~0.73

What's new vs simple:
  - Network depth: 3 conv layers -> 4 conv layers (64x64 -> 4x4)
  - Channels widened: 12/24/48 -> 32/64/128/256
  - BatchNorm after every conv (stabler training, can use larger lr)
  - LeakyReLU(0.2) instead of ReLU (no dead neurons)
  - Explicit FC bottleneck with latent_dim=256
  - Training: 30 -> 80 epochs, cosine annealing lr schedule
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'medium_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'medium_model.pt')

BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
LATENT_DIM = 256
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


class MediumAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),      # 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),     # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),    # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),   # 8 -> 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder_conv = nn.Sequential(
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
        h = self.encoder_conv(x)
        z = self.fc_enc(h.flatten(1))
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder_conv(h)


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

    model = MediumAE().to(device)
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
