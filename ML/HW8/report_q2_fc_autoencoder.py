"""
Report Q2 — Fully Connected Autoencoder + Latent Manipulation

Architecture (matches reference notebook fcn_autoencoder):
  Encoder: 64*64*3 (12288) -> 128 -> 64 -> 12 -> 3   [latent dim = 3]
  Decoder: 3 -> 12 -> 64 -> 128 -> 12288 -> reshape (3, 64, 64)

After training, we:
  1. Pick a sample face image
  2. Encode to get latent z (3-dim)
  3. Reconstruct (baseline)
  4. Reconstruct with z[0] *= 2 (scale first latent dim)
  5. Reconstruct with z[1] *= 2 (scale second latent dim)
  6. Reconstruct with z[2] *= 2 (scale third latent dim)
  7. Plot original + 4 reconstructions side-by-side

Output: output/q2_latent_manipulation.png
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'q2_fc_ae_model.pt')
FIG_PATH = os.path.join(OUTPUT_DIR, 'q2_latent_manipulation.png')

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
        self.data = np.load(npy_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1)


class FCAutoencoder(nn.Module):
    """Fully connected autoencoder. Latent dim = 3 (very tight bottleneck)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64),          nn.ReLU(inplace=True),
            nn.Linear(64, 12),           nn.ReLU(inplace=True),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),            nn.ReLU(inplace=True),
            nn.Linear(12, 64),           nn.ReLU(inplace=True),
            nn.Linear(64, 128),          nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 64 * 3), nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x.reshape(x.size(0), -1))

    def decode(self, z):
        return self.decoder(z).reshape(-1, 3, 64, 64)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def train(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x in tqdm(loader, leave=False, desc=f'Epoch {epoch}/{EPOCHS}'):
            x = x.to(device, non_blocking=True)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch:2d}/{EPOCHS} | loss {total / len(loader):.6f}')


def to_image(x_chw):
    """[-1, 1] CHW tensor -> [0, 1] HWC numpy."""
    img = x_chw.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img + 1.0) / 2.0
    return np.clip(img, 0.0, 1.0)


def visualize_latent_manipulation(model, sample_x):
    """Encode sample, modify latent dims one-by-one, reconstruct, plot."""
    model.eval()
    with torch.no_grad():
        x = sample_x.unsqueeze(0).to(device)        # (1, 3, 64, 64)
        z = model.encode(x).squeeze(0)              # (3,)
        recon_base = model.decode(z.unsqueeze(0)).squeeze(0)

        manipulations = []
        for dim in range(3):
            z_mod = z.clone()
            z_mod[dim] = z_mod[dim] * 2.0
            recon_mod = model.decode(z_mod.unsqueeze(0)).squeeze(0)
            manipulations.append((dim, z[dim].item(), z_mod[dim].item(), recon_mod))

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2))

    axes[0].imshow(to_image(sample_x))
    axes[0].set_title(f'Original\n(input image)', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(to_image(recon_base))
    axes[1].set_title(f'Reconstructed\n(z unchanged)\nz=[{z[0]:.2f},{z[1]:.2f},{z[2]:.2f}]', fontsize=10)
    axes[1].axis('off')

    for i, (dim, orig_val, new_val, recon) in enumerate(manipulations):
        ax = axes[2 + i]
        ax.imshow(to_image(recon))
        ax.set_title(f'z[{dim}] x 2\n({orig_val:.2f} -> {new_val:.2f})', fontsize=10)
        ax.axis('off')

    plt.suptitle('Q2 — FC Autoencoder: Effect of Scaling Each Latent Dimension',
                 fontsize=13, y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(FIG_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved figure -> {FIG_PATH}')

    return z, manipulations


def main():
    print(f'Device: {device}')

    train_loader = DataLoader(
        FaceDataset(os.path.join(DATA_DIR, 'trainingset.npy')),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )

    model = FCAutoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    if os.path.exists(MODEL_PATH):
        print(f'Loading existing model from {MODEL_PATH}')
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f'\n=== Training FC autoencoder ({EPOCHS} epochs) ===')
        train(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'Saved model -> {MODEL_PATH}')

    print('\n=== Generating Q2 visualization ===')
    test_data = np.load(os.path.join(DATA_DIR, 'testingset.npy'))
    sample_idx = 0
    x = test_data[sample_idx].astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    sample = torch.from_numpy(x).permute(2, 0, 1)
    print(f'Using test image index {sample_idx}')

    z, _ = visualize_latent_manipulation(model, sample)
    print(f'Original latent z = {z.cpu().numpy()}')


if __name__ == '__main__':
    main()
