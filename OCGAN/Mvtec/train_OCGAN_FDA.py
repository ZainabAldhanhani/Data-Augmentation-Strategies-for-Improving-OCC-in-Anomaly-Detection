import argparse
import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default image transform
def get_default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

default_transform = get_default_transform()

# Fourier Domain Adaptation (FDA) augmentation class
class FDATransform:
    def __init__(self, root, patch_ratio=0.1, probability=0.5):
        self.probability = probability
        self.patch_ratio = patch_ratio
        self.src_images = []
        # collect training source images
        train_dir = os.path.join(root, 'train', 'good')
        for ext in ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'):
            self.src_images += glob.glob(os.path.join(train_dir, ext))

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability or not self.src_images:
            return img
        tgt = np.array(img).astype(np.float32)
        src_path = random.choice(self.src_images)
        src = np.array(Image.open(src_path).convert('L').resize(img.size)).astype(np.float32)
        # FFT
        fft_tgt = np.fft.fft2(tgt)
        fft_src = np.fft.fft2(src)
        amp_tgt, pha_tgt = np.abs(fft_tgt), np.angle(fft_tgt)
        amp_src = np.abs(fft_src)
        h, w = tgt.shape
        b = int(min(h, w) * self.patch_ratio)
        cy, cx = h // 2, w // 2
        # swap low-frequency patch
        amp_tgt[cy-b:cy+b, cx-b:cx+b] = amp_src[cy-b:cy+b, cx-b:cx+b]
        fft_new = amp_tgt * np.exp(1j * pha_tgt)
        img_back = np.fft.ifft2(fft_new)
        img_back = np.real(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        return Image.fromarray(img_back)

# 1. Dataset loader for MVTec AD
class MVTecDataset(Dataset):
    def __init__(self, root, phase='train', transform=None):
        self.phase = phase
        self.transform = transform or default_transform
        self.images, self.labels = [], []
        if phase == 'train':
            good_dir = os.path.join(root, 'train', 'good')
            for img_path in sorted(glob.glob(os.path.join(good_dir, '*.png'))):
                self.images.append(img_path)
        else:
            test_root = os.path.join(root, 'test')
            for cls in sorted(os.listdir(test_root)):
                cls_dir = os.path.join(test_root, cls)
                if not os.path.isdir(cls_dir):
                    continue
                lbl = 0 if cls == 'good' else 1
                for img_path in sorted(glob.glob(os.path.join(cls_dir, '*.png'))):
                    self.images.append(img_path)
                    self.labels.append(lbl)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        x = self.transform(img)
        if self.phase != 'train':
            return x, self.labels[idx]
        return x

# 2. OCGAN network components
LATENT_DIM = 128

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128*16*16, LATENT_DIM)
        )
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 128*16*16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(-1,128,16,16)
        return self.deconv(x)

class DiscriminatorZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM,256), nn.LeakyReLU(0.2),
            nn.Linear(256,1), nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

class DiscriminatorX(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128*16*16,1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 3. OCGAN training with FDA transformation
def train_ocgan_mvtec(root, epochs=20, batch_size=32, lr=2e-4):
    train_tf = transforms.Compose([
        FDATransform(root, patch_ratio=0.1, probability=0.5),
        get_default_transform()
    ])
    ds = MVTecDataset(root, phase='train', transform=train_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    E, G = Encoder().to(device), Generator().to(device)
    D_z, D_x = DiscriminatorZ().to(device), DiscriminatorX().to(device)
    optE = optim.Adam(E.parameters(), lr=lr)
    optG = optim.Adam(G.parameters(), lr=lr)
    optDz = optim.Adam(D_z.parameters(), lr=lr)
    optDx = optim.Adam(D_x.parameters(), lr=lr)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    for ep in range(1, epochs+1):
        for x in loader:
            x = x.to(device)
            b = x.size(0)
            z_real = E(x)
            z_prior = torch.randn(b, LATENT_DIM, device=device)
            x_rec = G(z_real)
            optDz.zero_grad()
            dz_r = D_z(z_prior)
            dz_f = D_z(z_real.detach())
            lossDz = bce(dz_r, torch.ones_like(dz_r)) + bce(dz_f, torch.zeros_like(dz_f))
            lossDz.backward(); optDz.step()
            optDx.zero_grad()
            dx_r = D_x(x)
            dx_f = D_x(x_rec.detach())
            lossDx = bce(dx_r, torch.ones_like(dx_r)) + bce(dx_f, torch.zeros_like(dx_f))
            lossDx.backward(); optDx.step()
            optE.zero_grad(); optG.zero_grad()
            lossEz = bce(D_z(z_real), torch.ones_like(dz_r))
            lossGx = bce(D_x(x_rec), torch.ones_like(dx_r))
            lossRec = mse(x_rec, x)
            loss = lossRec + 0.1 * (lossEz + lossGx)
            loss.backward(); optE.step(); optG.step()
        print(f"Epoch {ep}/{epochs}: Dz={lossDz.item():.4f}, Dx={lossDx.item():.4f}, EG={loss.item():.4f}")

    torch.save(E.state_dict(), 'mvtec_ocgan_E.pth')
    torch.save(G.state_dict(), 'mvtec_ocgan_G.pth')
    return E, G

# 4. Evaluation function
def evaluate_ocgan_mvtec(E, G, root, batch_size=32):
    ds = MVTecDataset(root, phase='test', transform=get_default_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = E(x)
            x_rec = G(z)
            err = torch.mean((x_rec - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist()); labels.extend(y)
    scores, labels = np.array(scores), np.array(labels)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores)
    thr = th[np.argmax(tpr - fpr)]
    preds = (scores >= thr).astype(int)
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nConfusion Matrix:\n{cm}")
    print(classification_report(labels, preds, target_names=['Good','Defective']))


def main(dataset_path):
    root = dataset_path
    E, G = train_ocgan_mvtec(root)
    evaluate_ocgan_mvtec(E, G, root)


if __name__ == '__main__':
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
