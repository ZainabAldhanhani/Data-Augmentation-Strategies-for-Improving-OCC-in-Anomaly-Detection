import argparse
import os
import glob
import re
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

# Default transform
def get_default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

default_transform = get_default_transform()

# Fourier Domain Adaptation (FDA) augmentation
class FDATransform:
    def __init__(self, root, patch_ratio=0.1, probability=0.5):
        self.probability = probability
        self.patch_ratio = patch_ratio
        # collect source frames for spectral mixing
        src_dir = os.path.join(root, 'Train')
        self.src_paths = []
        for vid in os.listdir(src_dir):
            vid_dir = os.path.join(src_dir, vid)
            if not os.path.isdir(vid_dir):
                continue
            for ext in ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'):
                self.src_paths += glob.glob(os.path.join(vid_dir, ext))

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability or not self.src_paths:
            return img
        tgt = np.array(img).astype(np.float32)
        src_path = random.choice(self.src_paths)
        src_img = Image.open(src_path).convert('L').resize(img.size)
        src = np.array(src_img).astype(np.float32)
        # FFT
        fft_tgt = np.fft.fft2(tgt)
        fft_src = np.fft.fft2(src)
        amp_tgt, pha_tgt = np.abs(fft_tgt), np.angle(fft_tgt)
        amp_src = np.abs(fft_src)
        # mix amplitude in central frequency patch
        h, w = tgt.shape
        b = int(min(h, w) * self.patch_ratio)
        cy, cx = h//2, w//2
        amp_tgt[cy-b:cy+b, cx-b:cx+b] = amp_src[cy-b:cy+b, cx-b:cx+b]
        # inverse FFT
        fft_new = amp_tgt * np.exp(1j * pha_tgt)
        img_back = np.fft.ifft2(fft_new)
        img_back = np.real(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        return Image.fromarray(img_back)

# 1. Load ground truth for Ped1
def load_ucsd_gt(root):
    test_dir = os.path.join(root, 'Test')
    m_files = glob.glob(os.path.join(test_dir, '*.m'))
    if not m_files:
        raise FileNotFoundError(f"No .m GT files in {test_dir}")
    text = open(m_files[0], 'r').read()
    entries = re.findall(r"TestVideoFile\{end\+1\}\.gt_frame\s*=\s*\[([^\]]+)\];", text)
    if not entries:
        raise ValueError("No gt_frame entries found in .m file.")
    gt_list = []
    for entry in entries:
        frames = []
        for part in entry.split(','):
            part = part.strip()
            if ':' in part:
                s, e = map(int, part.split(':'))
                frames.extend(range(s, e+1))
            elif part.isdigit():
                frames.append(int(part))
        gt_list.append(frames)
    return gt_list

# 2. Dataset loader for UCSD Ped1
class UCSDPed1Dataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or default_transform
        subdir = 'Train' if phase == 'training' else 'Test'
        base = os.path.join(root, subdir)
        vids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        self.paths, self.labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    try:
                        with Image.open(p) as img:
                            img.convert('L'); img.load()
                    except:
                        continue
                    if phase == 'testing' and gt_list:
                        idx = int(re.sub(r'[^0-9]', '', vid)) - 1
                        fid = int(os.path.splitext(os.path.basename(p))[0])
                        label = 1 if fid in gt_list[idx] else 0
                    else:
                        label = 0
                    self.paths.append(p)
                    self.labels.append(label)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase == 'testing' else x

# 3. OCGAN components
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
    def forward(self, x): return self.net(x)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM,128*16*16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Tanh()
        )
    def forward(self, z): return self.deconv(self.fc(z).view(-1,128,16,16))
class DiscriminatorZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM,256), nn.LeakyReLU(0.2),
            nn.Linear(256,1), nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)
class DiscriminatorX(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128*16*16,1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 4. Training loop for Ped1 with FDA
def train_ocgan_ped1(root, epochs=20, batch_size=32, lr=2e-4):
    gt_list = load_ucsd_gt(root)
    train_tf = transforms.Compose([
        FDATransform(root, patch_ratio=0.1, probability=0.5),
        default_transform
    ])
    ds = UCSDPed1Dataset(root, 'training', transform=train_tf, gt_list=gt_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    E, G = Encoder().to(device), Generator().to(device)
    D_z, D_x = DiscriminatorZ().to(device), DiscriminatorX().to(device)
    optE, optG = optim.Adam(E.parameters(), lr=lr), optim.Adam(G.parameters(), lr=lr)
    optDz, optDx = optim.Adam(D_z.parameters(), lr=lr), optim.Adam(D_x.parameters(), lr=lr)
    bce, mse = nn.BCELoss(), nn.MSELoss()
    for ep in range(1, epochs+1):
        for x in loader:
            x = x.to(device); b = x.size(0)
            z_real = E(x)
            z_prior = torch.randn(b, LATENT_DIM, device=device)
            x_rec = G(z_real)
            optDz.zero_grad()
            dz_real = D_z(z_prior)
            dz_fake = D_z(z_real.detach())
            lossDz = bce(dz_real, torch.ones_like(dz_real)) + bce(dz_fake, torch.zeros_like(dz_fake))
            lossDz.backward(); optDz.step()
            optDx.zero_grad()
            dx_real = D_x(x)
            dx_fake = D_x(x_rec.detach())
            lossDx = bce(dx_real, torch.ones_like(dx_real)) + bce(dx_fake, torch.zeros_like(dx_fake))
            lossDx.backward(); optDx.step()
            optE.zero_grad(); optG.zero_grad()
            lossEz = bce(D_z(z_real), torch.ones_like(dz_real))
            lossGx = bce(D_x(x_rec), torch.ones_like(dx_real))
            lossRec = mse(x_rec, x)
            loss = lossRec + 0.1*(lossEz + lossGx)
            loss.backward(); optE.step(); optG.step()
        print(f"Epoch {ep}/{epochs}: Dz={lossDz.item():.4f}, Dx={lossDx.item():.4f}, EG={loss.item():.4f}")
    torch.save(E.state_dict(), 'ped1_ocgan_E.pth')
    torch.save(G.state_dict(), 'ped1_ocgan_G.pth')
    return E, G, gt_list

# 5. Evaluation for Ped1
def evaluate_ocgan_ped1(E, G, root, gt_list, batch_size=32):
    E.eval(); G.eval()
    ds = UCSDPed1Dataset(root, 'testing', transform=default_transform, gt_list=gt_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            try:
                x = x.to(device)
            except:
                continue
            x_rec = G(E(x))
            err = torch.mean((x_rec - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist()); labels.extend(y)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, ths = roc_curve(labels, scores)
    thr = ths[np.argmax(tpr-fpr)]
    preds = [1 if s>=thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds); f1 = f1_score(labels, preds)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nConfusion Matrix:\n{cm}")
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))


def main(dataset_path):
    root = dataset_path
    E, G, gt_list = train_ocgan_ped1(root)
    evaluate_ocgan_ped1(E, G, root, gt_list)


if __name__ == '__main__':
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
