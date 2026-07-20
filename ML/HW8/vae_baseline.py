"""
HW8 VAE Baseline — Variational Autoencoder
Inspired by reference notebook (ML2023_HW08.ipynb) which uses VAE as one of
its model choices.

What's new vs medium:
  - Encoder outputs mu and logvar (two LATENT_DIM vectors), not a single z
  - Reparameterization trick: z = mu + exp(0.5*logvar) * epsilon
  - Loss = MSE(recon, x) + beta * KL_divergence(q(z|x) || N(0, I))
  - KL regularization keeps the latent space well-behaved (close to N(0, I)),
    which means the encoder can't just memorize training samples in arbitrary
    latent positions — it has to compress face structure into a meaningful
    distribution. Anomalies that don't fit the learned distribution get
    poorly reconstructed.
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'vae_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'vae_model.pt')

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 256
KL_BETA = 1e-3       # KL weight: low so reconstruction dominates
WARMUP_EPOCHS = 3    # cosine with warmup, matching reference notebook
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


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=KL_BETA):
    recon_loss = F.mse_loss(recon, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss.item(), kld.item()


def get_lr(epoch, total, warmup):
    """Cosine annealing with linear warmup."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train_one_epoch(model, loader, optimizer, lr_factor):
    model.train()
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lr_factor

    total_recon = 0.0
    total_kld = 0.0
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        recon, mu, logvar = model(x)
        loss, r, k = vae_loss(recon, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_recon += r
        total_kld += k
    n = len(loader)
    return total_recon / n, total_kld / n


@torch.no_grad()
def compute_anomaly_scores(model, loader):
    """Use mean of the encoder distribution (deterministic) for inference."""
    model.eval()
    scores = []
    for x in tqdm(loader, desc='Inference', leave=False):
        x = x.to(device, non_blocking=True)
        mu, logvar = model.encode(x)
        recon = model.decode(mu)  # use mean directly, no sampling
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

    model = VAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('\n=== Training ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS, WARMUP_EPOCHS)
        recon_loss, kld_loss = train_one_epoch(model, train_loader, optimizer, lr_factor)
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | recon {recon_loss:.6f} | kld {kld_loss:.4f} | lr {LR * lr_factor:.2e}')

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
