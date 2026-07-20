"""
HW8 VQVAE Baseline — Vector Quantized VAE
Paper: van den Oord et al., "Neural Discrete Representation Learning" (NIPS 2017)
       https://arxiv.org/abs/1711.00937

Core idea (HARD quantization, vs MemAE's soft attention):
  Encoder outputs continuous z. A learnable codebook holds K embeddings e_1..e_K.
  Each spatial position of z is QUANTIZED to its NEAREST codebook entry e_k*.
  Decoder reconstructs from the quantized z (= e_k* at each position).

  Loss = reconstruction + codebook_loss + commitment_loss
  - codebook_loss: push e_k* toward encoder output (codebook learns prototypes)
  - commitment_loss: push encoder output toward codebook (encoder commits)
  - straight-through estimator: gradient bypasses quantization in backward

For anomaly detection:
  - Normal faces: encoder outputs fall near learned codebook entries -> small recon error
  - Anomalies (anime): encoder outputs fall in gaps between codebook entries
    -> quantized version is "wrong" face prototype -> large recon error
  - Hard quantization is more rigid than MemAE's soft averaging — different
    inductive bias, possibly different anomaly signal.
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'vqvae_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'vqvae_model.pt')

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
NUM_EMBEDDINGS = 128       # codebook size K
EMBEDDING_DIM = 256        # same as encoder output channels
COMMITMENT_COST = 0.25     # paper default
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


class VectorQuantizer(nn.Module):
    """Per-spatial-position vector quantization with straight-through estimator."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        # Paper init: uniform in [-1/K, 1/K]
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # inputs: (B, D, H, W) -> reshape to (B*H*W, D)
        b, d, h, w = inputs.shape
        flat = inputs.permute(0, 2, 3, 1).reshape(-1, d)

        # Squared distances: ||z||^2 + ||e||^2 - 2 z.e
        z2 = (flat ** 2).sum(dim=-1, keepdim=True)              # (B*H*W, 1)
        e2 = (self.embeddings.weight ** 2).sum(dim=-1)          # (K,)
        ze = flat @ self.embeddings.weight.t()                  # (B*H*W, K)
        distances = z2 + e2.unsqueeze(0) - 2 * ze               # (B*H*W, K)

        # Nearest codebook entry
        encoding_indices = distances.argmin(dim=-1)             # (B*H*W,)
        quantized = self.embeddings(encoding_indices)           # (B*H*W, D)

        # VQ losses
        codebook_loss = F.mse_loss(quantized, flat.detach())    # train codebook toward encoder
        commitment_loss = F.mse_loss(quantized.detach(), flat)  # train encoder toward codebook
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: forward = quantized, backward = identity to inputs
        quantized = flat + (quantized - flat).detach()

        # Reshape back
        quantized = quantized.reshape(b, h, w, d).permute(0, 3, 1, 2)
        return quantized, vq_loss, encoding_indices.reshape(b, h, w)


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=NUM_EMBEDDINGS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )  # 64 -> 4
        self.vq = VectorQuantizer(num_embeddings, EMBEDDING_DIM, COMMITMENT_COST)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )  # 4 -> 64

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices


def get_lr(epoch, total, warmup):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train_one_epoch(model, loader, optimizer, lr_factor):
    model.train()
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lr_factor

    total_recon = 0.0
    total_vq = 0.0
    n_unique_codes = []
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        recon, vq_loss, indices = model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_recon += recon_loss.item()
        total_vq += vq_loss.item()
        n_unique_codes.append(indices.unique().numel())
    n = len(loader)
    return total_recon / n, total_vq / n, float(np.mean(n_unique_codes))


@torch.no_grad()
def compute_anomaly_scores(model, loader):
    model.eval()
    scores = []
    for x in tqdm(loader, desc='Inference', leave=False):
        x = x.to(device, non_blocking=True)
        recon, _, _ = model(x)
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

    model = VQVAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}  (K={NUM_EMBEDDINGS} codebook x D={EMBEDDING_DIM})')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('\n=== Training ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS, WARMUP_EPOCHS)
        recon_loss, vq_loss, unique = train_one_epoch(model, train_loader, optimizer, lr_factor)
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | recon {recon_loss:.6f} | vq {vq_loss:.4f} | '
              f'unique_codes/batch {unique:.1f}/{NUM_EMBEDDINGS} | lr {LR * lr_factor:.2e}')

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
