import argparse
import os
import glob
import random
import re
import numpy as np
import scipy.io as sio
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score,
    f1_score, classification_report
)
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simulated Environmental Effects Augmentation ---
class SimulatedEnvTransform:
    def __init__(self,
                 brightness_range=(0.8, 1.2),
                 blur_radius=1.5,
                 noise_std=0.05,
                 probability=0.5):
        self.brightness_range = brightness_range
        self.blur_radius = blur_radius
        self.noise_std = noise_std
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img

        # 1) Random brightness
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(*self.brightness_range)
        )

        # 2) Random Gaussian blur
        img = img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0, self.blur_radius))
        )

        # 3) Additive Gaussian noise
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape) * self.noise_std
        arr = np.clip(arr + noise, 0, 1) * 255
        return Image.fromarray(arr.astype(np.uint8))

# --- Default transform (no augmentation) ---
def get_default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# --- 1. Avenue Dataset Loader ---
class AvenueDataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or get_default_transform()
        base = os.path.join(root, phase, 'frames')
        vids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        self.paths, self.labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    self.paths.append(p)
                    if phase == 'testing':
                        fid = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(vid) - 1
                        self.labels.append(1 if fid in gt_list[vid_idx] else 0)
                    else:
                        self.labels.append(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase == 'testing' else x

# --- 2. Load Avenue GT ---
def load_avenue_gt(root):
    mat = sio.loadmat(os.path.join(root, 'avenue.mat'))
    gt_cell = mat.get('gt', mat.get('gt_frame', None))[0]
    return [list(map(int, arr.flatten())) for arr in gt_cell]

# --- 3. OCGAN Components ---
LATENT_DIM = 128

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*16*16, LATENT_DIM)
        )
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 128*16*16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(-1,128,16,16)
        return self.net(x)

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
            nn.Flatten(),
            nn.Linear(128*16*16,1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# --- 4. OCGAN Training with SimulatedEnvTransform ---
def train_ocgan(root, epochs=50, batch_size=32, lr=2e-4):
    gt_list = load_avenue_gt(root)

    train_tf = transforms.Compose([
        SimulatedEnvTransform(
            brightness_range=(0.8,1.2),
            blur_radius=1.5,
            noise_std=0.05,
            probability=0.5
        ),
        get_default_transform()
    ])

    ds = AvenueDataset(root, 'training', transform=train_tf, gt_list=gt_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    E, G = Encoder().to(device), Generator().to(device)
    D_z, D_x = DiscriminatorZ().to(device), DiscriminatorX().to(device)

    opt_E = optim.Adam(E.parameters(), lr=lr)
    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_Dz = optim.Adam(D_z.parameters(), lr=lr)
    opt_Dx = optim.Adam(D_x.parameters(), lr=lr)

    bce, mse = nn.BCELoss(), nn.MSELoss()

    for ep in range(1, epochs+1):
        for x in loader:
            x = x.to(device)
            b = x.size(0)

            # encode & reconstruct
            z_real = E(x)
            z_prior = torch.randn(b, LATENT_DIM, device=device)
            x_rec = G(z_real)

            # update D_z
            opt_Dz.zero_grad()
            Dz_r = D_z(z_prior)
            Dz_f = D_z(z_real.detach())
            loss_Dz = bce(Dz_r, torch.ones_like(Dz_r)) \
                   + bce(Dz_f, torch.zeros_like(Dz_f))
            loss_Dz.backward()
            opt_Dz.step()

            # update D_x
            opt_Dx.zero_grad()
            Dx_r = D_x(x)
            Dx_f = D_x(x_rec.detach())
            loss_Dx = bce(Dx_r, torch.ones_like(Dx_r)) \
                   + bce(Dx_f, torch.zeros_like(Dx_f))
            loss_Dx.backward()
            opt_Dx.step()

            # update E & G
            loss_Ez = bce(D_z(z_real), torch.ones_like(Dz_r))
            loss_Gx = bce(D_x(x_rec), torch.ones_like(Dx_r))
            loss_rec = mse(x_rec, x)
            loss_total = loss_rec + 0.1*(loss_Ez + loss_Gx)

            opt_E.zero_grad()
            opt_G.zero_grad()
            loss_total.backward()
            opt_E.step()
            opt_G.step()

        print(f"Epoch {ep}/{epochs}: "
              f"Dz={loss_Dz.item():.4f}  "
              f"Dx={loss_Dx.item():.4f}  "
              f"EG={loss_total.item():.4f}")

    torch.save(E.state_dict(), 'avenue_ocgan_E.pth')
    torch.save(G.state_dict(), 'avenue_ocgan_G.pth')
    return E, G, gt_list

# --- 5. Evaluation ---
def evaluate_ocgan(E, G, root, gt_list, batch_size=32):
    E.eval(); G.eval()
    ds = AvenueDataset(root, 'testing', transform=get_default_transform(), gt_list=gt_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_rec = G(E(x))
            err = torch.mean((x_rec - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist())
            labels.extend(y)

    scores = np.array(scores)
    labels = np.array(labels)

    auc = roc_auc_score(labels, scores)
    fpr, tpr, ths = roc_curve(labels, scores)
    thr = ths[np.argmax(tpr - fpr)]
    preds = (scores >= thr).astype(int)

    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))


def main(dataset_path):
    root = dataset_path
    E, G, gt = train_ocgan(root, epochs=50, batch_size=32, lr=2e-4)
    evaluate_ocgan(E, G, root, gt, batch_size=32)


if __name__ == '__main__':
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
