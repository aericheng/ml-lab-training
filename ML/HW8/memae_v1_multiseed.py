"""
HW8 MemAE v1 multi-seed runner

Trains the "v1" version of MemAE (which empirically beats v2 — see
README section on the v1 vs v2 finding) with multiple random seeds.
Each independently-trained model gets ensembled at the end.

v1 config (the empirically-best, paper-broken-but-effective version):
  - Memory init: N(0, 0.01)        — small, soft routing
  - SHRINK_THRES: 0.0008           — below uniform, doesn't actually shrink
  - ENTROPY_WEIGHT: 2e-4           — paper value
  - NO softmax temperature scaling — keeps attention near-uniform

This gives "averaging-style" memory addressing which acts as a natural
anomaly filter (all queries get average-prototype-latent; anomalies can't
escape it).
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'output'

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 256
MEM_SIZE = 500
SHRINK_THRES = 0.0008      # v1: below 1/N, ineffective (intentional — keeps uniform attention)
ENTROPY_WEIGHT = 2e-4
WARMUP_EPOCHS = 3

SEEDS = [123, 2024]        # already have seed=42 saved as memae_v1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def hard_shrink_relu(x, lambd=0.0025, eps=1e-12):
    return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + eps)


class MemoryModuleV1(nn.Module):
    """v1: small N(0, 0.01) init, no softmax temperature -> near-uniform attention."""
    def __init__(self, mem_size, fea_dim, shrink_thres=0.0008):
        super().__init__()
        self.mem_size = mem_size
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.randn(mem_size, fea_dim) * 0.01)

    def forward(self, query):
        q_norm = F.normalize(query, dim=-1)
        m_norm = F.normalize(self.memory, dim=-1)
        att = F.linear(q_norm, m_norm)             # cosine sim
        att = F.softmax(att, dim=-1)               # no temperature
        if self.shrink_thres > 0:
            att = hard_shrink_relu(att, lambd=self.shrink_thres)
            att = F.normalize(att, p=1, dim=-1)
        z_hat = att @ self.memory
        return z_hat, att


class MemAE_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 4 * 4, LATENT_DIM)
        self.memory = MemoryModuleV1(MEM_SIZE, LATENT_DIM, SHRINK_THRES)
        self.fc_dec = nn.Linear(LATENT_DIM, 256 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x):
        h = self.encoder_conv(x).flatten(1)
        z = self.fc_enc(h)
        z_hat, att = self.memory(z)
        h_dec = self.fc_dec(z_hat).view(-1, 256, 4, 4)
        return self.decoder_conv(h_dec), att


def get_lr(epoch, total, warmup):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def memae_loss(recon, x, att):
    recon_loss = F.mse_loss(recon, x)
    eps = 1e-12
    entropy = -torch.mean(torch.sum(att * torch.log(att + eps), dim=-1))
    return recon_loss + ENTROPY_WEIGHT * entropy, recon_loss.item(), entropy.item()


def train_seed(seed, train_loader, test_loader):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MemAE_V1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f'\n=== Training MemAE v1 seed={seed} ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = LR * lr_factor

        model.train()
        total_recon = 0.0
        for x in tqdm(train_loader, leave=False, desc=f'seed={seed} ep {epoch+1}'):
            x = x.to(device, non_blocking=True)
            recon, att = model(x)
            loss, r, e = memae_loss(recon, x, att)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += r
        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f'  seed={seed} epoch {epoch+1:3d}/{EPOCHS} | recon {total_recon / len(train_loader):.6f}')

    # Inference
    model.eval()
    scores = []
    with torch.no_grad():
        for x in tqdm(test_loader, desc=f'seed={seed} infer', leave=False):
            x = x.to(device, non_blocking=True)
            recon, _ = model(x)
            err = ((recon - x) ** 2).mean(dim=[1, 2, 3])
            scores.extend(err.cpu().tolist())

    # Save
    csv_path = os.path.join(OUTPUT_DIR, f'memae_v1_seed{seed}_submission.csv')
    model_path = os.path.join(OUTPUT_DIR, f'memae_v1_seed{seed}_model.pt')
    with open(csv_path, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    torch.save(model.state_dict(), model_path)
    print(f'  saved {csv_path}, {model_path}')


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

    for seed in SEEDS:
        train_seed(seed, train_loader, test_loader)


if __name__ == '__main__':
    main()
