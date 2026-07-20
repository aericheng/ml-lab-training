"""
HW8 Spatial MemAE Baseline — MemAE with per-spatial-location memory addressing
Paper: Gong et al., "Memorizing Normality to Detect Anomaly" (ICCV 2019)
       https://arxiv.org/abs/1904.02639

This is the paper's actual recommended design (vs the flat-latent MemAE v1 we
already trained):

  After encoder: feature map of shape (B, 256, 4, 4) = 16 spatial positions
  For EACH of the 16 positions, query the memory independently:
    -> 16x more attention queries per image
    -> finer-grained "prototype matching" per local region
  Memory items are 256-dim vectors (one per channel-vector across spatial).

Uses v1's empirically-good config (which beats v2's paper-faithful sparse version):
  - No softmax temperature scaling
  - Small memory init N(0, 0.01)
  - SHRINK_THRES below 1/N (effectively disabled)
  - The resulting "uniform-ish soft attention" acts as a natural averaging
    bottleneck (see README section 7.11 for the v1 vs v2 reversal story)
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'spatial_memae_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'spatial_memae_model.pt')

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
MEM_SIZE = 500            # number of prototypes (each 256-dim)
FEA_DIM = 256             # channel dim of feature map (must match encoder output)
SHRINK_THRES = 0.0008     # v1 "broken-but-good" config: below 1/N, effectively disabled
ENTROPY_WEIGHT = 2e-4
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
    return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + eps)


class SpatialMemoryModule(nn.Module):
    """Apply memory addressing to each spatial position of a (B, C, H, W) feature map.
    Reshapes to (B*H*W, C), queries N memory items, reshapes back.
    """
    def __init__(self, mem_size, fea_dim, shrink_thres=0.0008):
        super().__init__()
        self.mem_size = mem_size
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        # Same v1 init
        self.memory = nn.Parameter(torch.randn(mem_size, fea_dim) * 0.01)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Treat each spatial position as a separate query
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        q_norm = F.normalize(x_flat, dim=-1)
        m_norm = F.normalize(self.memory, dim=-1)
        att = F.linear(q_norm, m_norm)              # (B*H*W, N) cosine sim
        att = F.softmax(att, dim=-1)                # no temperature (v1 style)

        if self.shrink_thres > 0:
            att = hard_shrink_relu(att, lambd=self.shrink_thres)
            att = F.normalize(att, p=1, dim=-1)

        z_hat_flat = att @ self.memory              # (B*H*W, C)
        # Reshape back to spatial
        z_hat = z_hat_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return z_hat, att


class SpatialMemAE(nn.Module):
    """Conv encoder -> Spatial memory -> Conv decoder.
    No FC bottleneck: memory operates directly on the spatial feature map.
    """
    def __init__(self, mem_size=MEM_SIZE, shrink_thres=SHRINK_THRES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )  # 64x64 -> 4x4
        self.memory = SpatialMemoryModule(mem_size, FEA_DIM, shrink_thres)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )  # 4x4 -> 64x64

    def forward(self, x):
        feat = self.encoder(x)          # (B, 256, 4, 4)
        z_hat, att = self.memory(feat)  # (B, 256, 4, 4), (B*16, 500)
        recon = self.decoder(z_hat)
        return recon, att


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

    model = SpatialMemAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}  (MEM_SIZE={MEM_SIZE}, FEA_DIM={FEA_DIM})')
    print(f'  -> {16 * MEM_SIZE:,} effective prototype slots per image (16 spatial * {MEM_SIZE} mem)')

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
