"""
Generate visualizations for HW8 Notion writeup.

Outputs PNG files to output/images/:
  01_training_data.png       ??5x5 grid of training faces
  02_test_data_random.png    ??5x5 grid of random test images
  03_recon_<model>.png       ??per-model reconstruction (low/high score samples)
  04_recon_compare_all.png   ??same input images, all 4 models' reconstructions
  05_score_distributions.png ??anomaly score histograms
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

HW8_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HW8_DIR, 'data')
OUT_DIR = os.path.join(HW8_DIR, 'output')
IMG_DIR = os.path.join(OUT_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 3, 4, 2, 1), nn.Tanh(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class MediumAE(nn.Module):
    def __init__(self, latent_dim=256):
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
    def forward(self, x):
        h = self.encoder_conv(x)
        z = self.fc_enc(h.flatten(1))
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder_conv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
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
    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder_conv(h)
    def forward(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)


def hard_shrink_relu(x, lambd=0.0025, eps=1e-12):
    return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + eps)


class MemoryModule(nn.Module):
    """MemAE v1 (broken-but-good) ??no softmax temperature, low shrink_thres
    ??uniform attention ??z_hat ??average prototype (the "bug" that breaks the
    bottleneck open and makes AUC actually higher).
    """
    def __init__(self, mem_size, fea_dim, shrink_thres=0.0008):
        super().__init__()
        self.mem_size = mem_size
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = nn.Parameter(torch.randn(mem_size, fea_dim) * 0.01)

    def forward(self, query):
        q_norm = F.normalize(query, dim=-1)
        m_norm = F.normalize(self.memory, dim=-1)
        att = F.linear(q_norm, m_norm)
        att = F.softmax(att, dim=-1)  # no temperature ??uniform attention emerges
        if self.shrink_thres > 0:
            att = hard_shrink_relu(att, lambd=self.shrink_thres)
            att = F.normalize(att, p=1, dim=-1)
        z_hat = att @ self.memory
        return z_hat, att


class MemAE(nn.Module):
    """MemAE ??same encoder/decoder as Medium, with Memory module between."""
    def __init__(self, latent_dim=256, mem_size=500, shrink_thres=0.0008):
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
        z_hat, _ = self.memory(z)
        h_dec = self.fc_dec(z_hat).view(-1, 256, 4, 4)
        return self.decoder_conv(h_dec)


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BasicBlock(64, 64, 2), BasicBlock(64, 64))
        self.stage2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.stage3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.stage4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc_enc = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )
    def encode(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        return self.fc_enc(x.flatten(1))
    def decode(self, z):
        h = self.fc_dec(z).view(-1, 512, 4, 4)
        return self.decoder(h)
    def forward(self, x):
        return self.decode(self.encode(x))


def preprocess(x_uint8):
    x = x_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def deprocess(x_tensor):
    x = x_tensor.cpu().detach()
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = x.permute(0, 2, 3, 1).numpy()
    return (x * 255).astype(np.uint8)


print('Loading data...')
train_data = np.load(os.path.join(DATA_DIR, 'trainingset.npy'), mmap_mode='r')
test_data = np.load(os.path.join(DATA_DIR, 'testingset.npy'))
print(f'  train: {train_data.shape}, test: {test_data.shape}')


# ===== IMG 1: Training data 5x5 grid =====
np.random.seed(42)
idx = np.random.choice(len(train_data), 25, replace=False)
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_data[idx[i]])
    ax.axis('off')
fig.suptitle('Training Data — 25 random real faces (100k total)', fontsize=14, y=0.93)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '01_training_data.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved 01_training_data.png')


# ===== IMG 2: Test data 5x5 random =====
np.random.seed(7)
idx = np.random.choice(len(test_data), 25, replace=False)
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_data[idx[i]])
    ax.axis('off')
fig.suptitle('Test Data — 25 random images (mix of normal + anomaly)', fontsize=14, y=0.93)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '02_test_data_random.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved 02_test_data_random.png')


print('\nLoading models and running inference...')
models = [
    ('simple', ConvAutoencoder(), 'simple_model.pt'),
    ('medium', MediumAE(), 'medium_model.pt'),
    ('vae', VAE(), 'vae_model.pt'),
    ('resnet', ResNetAE(), 'resnet_model.pt'),
    ('memae', MemAE(), 'memae_v1_model.pt'),
]

test_tensor = preprocess(np.array(test_data))
scores = {}
recons = {}

for name, model, weight_file in models:
    state = torch.load(os.path.join(OUT_DIR, weight_file), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    BATCH = 256
    all_recons, all_scores = [], []
    with torch.no_grad():
        for i in range(0, len(test_tensor), BATCH):
            x = test_tensor[i:i + BATCH].to(DEVICE, non_blocking=True)
            recon = model(x)
            err = ((recon - x) ** 2).mean(dim=[1, 2, 3])
            all_recons.append(recon.cpu())
            all_scores.append(err.cpu())
    recons[name] = torch.cat(all_recons)
    scores[name] = torch.cat(all_scores).numpy()
    print(f'  {name}: score range=[{scores[name].min():.4f}, {scores[name].max():.4f}], mean={scores[name].mean():.4f}, std={scores[name].std():.4f}')

    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


# ===== IMG 3a-d: per-model reconstruction =====
for name in ['simple', 'medium', 'vae', 'resnet', 'memae']:
    s = scores[name]
    low_idx = np.argsort(s)[:3]                # 3 lowest score ??"predicted normal"
    high_idx = np.argsort(s)[::-1][:3]         # 3 highest score ??"predicted anomaly"
    indices = list(low_idx) + list(high_idx)

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    for col, idx in enumerate(indices):
        axes[0, col].imshow(test_data[idx])
        axes[0, col].axis('off')
        tag = 'normal-like' if col < 3 else 'anomaly-like'
        axes[0, col].set_title(f'#{idx}\n{tag}\nscore={s[idx]:.4f}', fontsize=9)
        recon_img = deprocess(recons[name][idx:idx + 1])[0]
        axes[1, col].imshow(recon_img)
        axes[1, col].axis('off')

    # Row labels anchored to each row's first axes (always aligned, never clipped)
    axes[0, 0].text(-0.22, 0.5, 'Input', transform=axes[0, 0].transAxes,
                    ha='right', va='center', fontsize=12, weight='bold')
    axes[1, 0].text(-0.22, 0.5, 'Recon', transform=axes[1, 0].transAxes,
                    ha='right', va='center', fontsize=12, weight='bold')
    fig.suptitle(f'{name.upper()} — Reconstruction of low/high-score test images', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    plt.savefig(os.path.join(IMG_DIR, f'03_recon_{name}.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved 03_recon_{name}.png')


# ===== IMG 4: All models compared on same test images =====
s_vae = scores['vae']
low_idx = np.argsort(s_vae)[:3]
high_idx = np.argsort(s_vae)[::-1][:3]
indices = list(low_idx) + list(high_idx)

n_rows = 6  # original + 5 models
fig, axes = plt.subplots(n_rows, 6, figsize=(15, 15.5))
row_labels = ['Original', 'simple', 'medium', 'vae', 'resnet', 'memae']

# Row 0: original
for col, idx in enumerate(indices):
    axes[0, col].imshow(test_data[idx])
    axes[0, col].axis('off')
    tag = 'normal-like' if col < 3 else 'anomaly-like'
    axes[0, col].set_title(f'#{idx} ({tag})\nscore_vae={s_vae[idx]:.4f}', fontsize=9)

# Rows 1-5: each model
for row, name in enumerate(['simple', 'medium', 'vae', 'resnet', 'memae'], start=1):
    for col, idx in enumerate(indices):
        recon_img = deprocess(recons[name][idx:idx + 1])[0]
        axes[row, col].imshow(recon_img)
        axes[row, col].axis('off')

# Row labels anchored to each row's first axes (always aligned with the row)
for row, label in enumerate(row_labels):
    axes[row, 0].text(-0.20, 0.5, label.upper(), transform=axes[row, 0].transAxes,
                      ha='right', va='center', fontsize=11, weight='bold')

fig.suptitle('All 5 Models — Reconstruction of Same Test Images (MemAE is the breakthrough)', fontsize=13, y=1.01)
plt.tight_layout(rect=[0.06, 0, 1, 0.97])
plt.savefig(os.path.join(IMG_DIR, '04_recon_compare_all.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved 04_recon_compare_all.png')


# ===== IMG 5: Score distributions =====
fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
colors = ['#888888', '#3674B5', '#3FA34D', '#FFAA00', '#E63946']
for ax, name, color in zip(axes, ['simple', 'medium', 'vae', 'resnet', 'memae'], colors):
    s = scores[name]
    ax.hist(s, bins=80, color=color, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_title(f'{name.upper()}\nstd={s.std():.4f}, max={s.max():.4f}', fontsize=12)
    ax.set_xlabel('anomaly score (MSE)')
    if ax is axes[0]:
        ax.set_ylabel('# test images')
    ax.grid(alpha=0.3)

fig.suptitle('Anomaly Score Distributions (per model)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(IMG_DIR, '05_score_distributions.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved 05_score_distributions.png')

print(f'\n??All images saved to: {IMG_DIR}')

