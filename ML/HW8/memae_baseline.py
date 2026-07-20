"""
HW8 MemAE Baseline v2 — Memory-Augmented Autoencoder (paper-faithful)
Paper: Gong et al., "Memorizing Normality to Detect Anomaly" (ICCV 2019)
       https://arxiv.org/abs/1904.02639

v2 changes from v1:
  - Memory init: U(-1/sqrt(d), 1/sqrt(d)) instead of N(0, 0.01)
    -> larger init = better cosine variance = sharper softmax from start
  - SHRINK_THRES: 0.0008 -> 0.005 (paper recommended range 1/N to 3/N for N=500)
    -> actually forces sparsity (was below uniform 1/N so never zeroed anything)
  - ENTROPY_WEIGHT: 2e-4 -> 2e-3 (stronger pressure toward sparse attention)

Expected effect: entropy should actually decrease during training,
                 memory items differentiate into distinct prototypes,
                 attention becomes truly sparse (only top few prototypes per query).
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'memae_v2_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'memae_v2_model.pt')

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 256
MEM_SIZE = 500            # number of prototype vectors (paper used 100-2000)
SHRINK_THRES = 0.005      # paper recommends 1/N to 3/N (i.e. 0.002 to 0.006 for N=500)
ENTROPY_WEIGHT = 2e-3     # weight on attention entropy loss (paper used 2e-4 but our N is larger)
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


def hard_shrink_relu(x, lambd=0.0025, eps=1e-12):
    """Smooth approximation of hard thresholding (from MemAE paper Eq. 7).
    Maps small values to 0 while keeping large values approximately unchanged.
    """
    return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + eps)


class MemoryModule(nn.Module):
    """N learnable prototype vectors + attention-based retrieval.

    Forward: given query z (B, D)
      1. Compute cosine-similarity attention to N prototypes -> (B, N)
      2. Softmax to get attention weights
      3. Hard shrinkage for sparsity (only top prototypes contribute)
      4. Renormalize, then weighted sum of prototypes -> z_hat (B, D)
    """
    def __init__(self, mem_size, fea_dim, shrink_thres=0.0025):
        super().__init__()
        self.mem_size = mem_size
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        # Paper init: U(-1/sqrt(d), 1/sqrt(d)) -- larger spread than N(0, 0.01)
        # so cosine similarities have meaningful variance from the start
        stdv = 1.0 / math.sqrt(fea_dim)
        self.memory = nn.Parameter(torch.empty(mem_size, fea_dim).uniform_(-stdv, stdv))

    def forward(self, query):
        # query: (B, D); memory: (N, D)
        q_norm = F.normalize(query, dim=-1)
        m_norm = F.normalize(self.memory, dim=-1)
        att = F.linear(q_norm, m_norm)              # (B, N) cosine similarities in [-1, 1]
        # Temperature scaling: softmax of values in [-1, 1] is too flat for N=500.
        # tau=10 sharpens it so a well-aligned prototype gets meaningful mass.
        att = F.softmax(att * 10.0, dim=-1)         # (B, N) weights

        if self.shrink_thres > 0:
            att = hard_shrink_relu(att, lambd=self.shrink_thres)
            # Renormalize so rows sum to 1
            att = F.normalize(att, p=1, dim=-1)

        z_hat = att @ self.memory                   # (B, D)
        return z_hat, att


class MemAE(nn.Module):
    """Conv encoder + FC bottleneck + Memory module + FC + Conv decoder.
    Same backbone as medium AE, with Memory module between latent and decoder.
    """
    def __init__(self, latent_dim=LATENT_DIM, mem_size=MEM_SIZE, shrink_thres=SHRINK_THRES):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        self.memory = MemoryModule(mem_size, latent_dim, shrink_thres)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
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
        recon = self.decoder_conv(h_dec)
        return recon, att


def get_lr(epoch, total, warmup):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def memae_loss(recon, x, att):
    """MSE reconstruction + entropy of attention (encourage sparsity)."""
    recon_loss = F.mse_loss(recon, x)
    # Entropy: -sum p log p over attention weights
    eps = 1e-12
    entropy = -torch.mean(torch.sum(att * torch.log(att + eps), dim=-1))
    return recon_loss + ENTROPY_WEIGHT * entropy, recon_loss.item(), entropy.item()


def train_one_epoch(model, loader, optimizer, lr_factor):
    model.train()
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lr_factor

    total_recon = 0.0
    total_ent = 0.0
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        recon, att = model(x)
        loss, r, e = memae_loss(recon, x, att)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_recon += r
        total_ent += e
    n = len(loader)
    return total_recon / n, total_ent / n


@torch.no_grad()
def compute_anomaly_scores(model, loader):
    model.eval()
    scores = []
    for x in tqdm(loader, desc='Inference', leave=False):
        x = x.to(device, non_blocking=True)
        recon, _ = model(x)
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

    model = MemAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_mem = model.memory.memory.numel()
    print(f'Model params: {n_params:,}  (memory: {n_mem:,} = {MEM_SIZE} prototypes x {LATENT_DIM} dim)')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('\n=== Training ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS, WARMUP_EPOCHS)
        recon_loss, ent_loss = train_one_epoch(model, train_loader, optimizer, lr_factor)
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | recon {recon_loss:.6f} | entropy {ent_loss:.4f} | lr {LR * lr_factor:.2e}')

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
