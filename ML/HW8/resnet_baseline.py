"""
HW8 ResNet Baseline — Residual encoder (from scratch, no pretrained)
PDF page 15 explicitly mentions "use Resnet as encoder" as a boss option.

What's new vs medium:
  - Encoder uses BasicBlock with skip connections (ResNet-style)
  - Skip connections let deeper networks train without degradation,
    capturing fine-grained face details that anomalies lack
  - Decoder kept as plain ConvTranspose (mirror)
  - Same MSE loss as medium
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'resnet_submission.csv')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'resnet_model.pt')

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 256
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


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs + skip connection."""
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
    """ResNet18-style encoder + plain ConvTranspose decoder."""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Stem: 3 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 4 stages, each 2 BasicBlocks, downsampling by 2 at stage start
        self.stage1 = nn.Sequential(BasicBlock(64, 64, 2), BasicBlock(64, 64))      # 64->32
        self.stage2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))   # 32->16
        self.stage3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))  # 16->8
        self.stage4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))  # 8->4
        self.fc_enc = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),  # 4->8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),  # 8->16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),    # 16->32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),     # 32->64
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.fc_enc(x.flatten(1))

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 512, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def get_lr(epoch, total, warmup):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def train_one_epoch(model, loader, optimizer, lr_factor):
    model.train()
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lr_factor

    total = 0.0
    for x in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        recon = model(x)
        loss = F.mse_loss(recon, x)
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

    model = ResNetAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('\n=== Training ===')
    for epoch in range(EPOCHS):
        lr_factor = get_lr(epoch, EPOCHS, WARMUP_EPOCHS)
        avg_loss = train_one_epoch(model, train_loader, optimizer, lr_factor)
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | loss {avg_loss:.6f} | lr {LR * lr_factor:.2e}')

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
