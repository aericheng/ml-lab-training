"""
HW8 Boss Baseline — Denoising AE + Classifier head
Target AUC: ~0.77+

What's new vs medium:
  1. Denoising training: add Gaussian noise to input, decoder must reconstruct
     the CLEAN target. Forces the encoder to throw away pixel noise and keep
     only structural / semantic face information -> more robust features,
     more sensitive to structural anomalies (anime faces).

  2. Classifier head on latent z: a small MLP that distinguishes
     "z(clean face)" vs "z(noisy face)". At inference, anomaly images
     (anime) look more like noisy faces from the latent's perspective,
     so the classifier provides an additional anomaly signal independent
     of the reconstruction error.

  3. Anomaly score = standardize(MSE) + standardize(1 - classifier_prob_real)
     Combining two orthogonal signals via z-score normalization avoids
     scale mismatch.

  Refs:
    - Denoising AE: Vincent et al. ICML 2008
    - DAE + classifier head: GANomaly / OCGAN family
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'boss_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'boss_model.pt')

BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
LATENT_DIM = 256
NOISE_SIGMA = 0.1      # std of Gaussian noise in normalized [-1, 1] space
CLS_WEIGHT = 0.1       # classifier loss weight (vs reconstruction)
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


class BossAE(nn.Module):
    """Conv AE (medium-style) + classifier head on latent."""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        return self.fc_enc(h.flatten(1))

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder_conv(h)

    def classify(self, z):
        return self.classifier(z).squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        logit = self.classify(z)
        return recon, z, logit


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_recon = 0.0
    total_cls = 0.0
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        # Add noise (in [-1, 1] space)
        noise = torch.randn_like(x) * NOISE_SIGMA
        x_noisy = (x + noise).clamp(-1.0, 1.0)

        # Forward both branches
        recon_clean, z_clean, logit_clean = model(x)
        recon_noisy, z_noisy, logit_noisy = model(x_noisy)

        # Denoising reconstruction: both should reconstruct the CLEAN x
        recon_loss = F.mse_loss(recon_clean, x) + F.mse_loss(recon_noisy, x)

        # Classifier: z(clean) -> 1 (real), z(noisy) -> 0 (corrupted)
        ones = torch.ones_like(logit_clean)
        zeros = torch.zeros_like(logit_noisy)
        cls_loss = F.binary_cross_entropy_with_logits(logit_clean, ones) + \
                   F.binary_cross_entropy_with_logits(logit_noisy, zeros)

        loss = recon_loss + CLS_WEIGHT * cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += recon_loss.item()
        total_cls += cls_loss.item()
    n = len(loader)
    return total_recon / n, total_cls / n


@torch.no_grad()
def compute_anomaly_scores(model, loader):
    """Combine reconstruction MSE + classifier "fake probability".
    Both standardized to z-scores before summing.
    """
    model.eval()
    mse_scores = []
    cls_scores = []
    for x in tqdm(loader, desc='Inference', leave=False):
        x = x.to(device, non_blocking=True)
        recon, z, logit = model(x)
        mse = ((recon - x) ** 2).mean(dim=[1, 2, 3])
        # 1 - sigmoid(logit) = probability of "fake/noisy" (higher = more anomalous)
        cls_fake = 1.0 - torch.sigmoid(logit)
        mse_scores.extend(mse.cpu().tolist())
        cls_scores.extend(cls_fake.cpu().tolist())

    mse_arr = np.array(mse_scores)
    cls_arr = np.array(cls_scores)
    mse_z = (mse_arr - mse_arr.mean()) / (mse_arr.std() + 1e-8)
    cls_z = (cls_arr - cls_arr.mean()) / (cls_arr.std() + 1e-8)
    return mse_z + cls_z, mse_arr, cls_arr


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

    model = BossAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print('\n=== Training ===')
    for epoch in range(1, EPOCHS + 1):
        recon_loss, cls_loss = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()
        print(f'Epoch {epoch:3d}/{EPOCHS} | recon {recon_loss:.6f} | cls {cls_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model -> {MODEL_PATH}')

    print('\n=== Inference ===')
    scores, mse_raw, cls_raw = compute_anomaly_scores(model, test_loader)
    print(f'mse range: [{mse_raw.min():.4f}, {mse_raw.max():.4f}]  mean={mse_raw.mean():.4f}')
    print(f'cls range: [{cls_raw.min():.4f}, {cls_raw.max():.4f}]  mean={cls_raw.mean():.4f}')

    with open(OUTPUT_CSV, 'w') as f:
        f.write('ID,score\n')
        for i, s in enumerate(scores):
            f.write(f'{i},{s}\n')
    print(f'Wrote {len(scores)} scores -> {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
