"""
HW8 Simple Baseline — Vanilla Conv Autoencoder
Target AUC: ~0.53
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'simple_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'simple_model.pt')

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FaceDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path)  # (N, 64, 64, 3) uint8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].astype(np.float32) / 255.0
        # Normalize to [-1, 1] so it matches the Tanh decoder output
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1),    # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),   # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),   # 16 -> 8
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


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

    model = ConvAutoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print('\n=== Training ===')
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f'Epoch {epoch:3d}/{EPOCHS} | loss {avg_loss:.6f}')

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
