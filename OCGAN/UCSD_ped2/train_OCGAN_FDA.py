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
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fourier Domain Adaptation (FDA) augmentation
class FDATransform:
    def __init__(self, root, patch_ratio=0.1, probability=0.5):
        self.probability = probability
        self.patch_ratio = patch_ratio
        self.src_paths = []
        src_dir = os.path.join(root, 'Train')
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
        fft_tgt = np.fft.fft2(tgt)
        fft_src = np.fft.fft2(src)
        amp_tgt, pha_tgt = np.abs(fft_tgt), np.angle(fft_tgt)
        amp_src = np.abs(fft_src)
        h, w = tgt.shape
        b = int(min(h, w) * self.patch_ratio)
        cy, cx = h//2, w//2
        amp_tgt[cy-b:cy+b, cx-b:cx+b] = amp_src[cy-b:cy+b, cx-b:cx+b]
        fft_new = amp_tgt * np.exp(1j * pha_tgt)
        img_back = np.fft.ifft2(fft_new)
        img_back = np.real(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        return Image.fromarray(img_back)

# Default transform (no augmentation)
def get_default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
# instantiate once
default_transform = get_default_transform()

# 1. Dataset loader for UCSD Ped2
default_transform  # use the instantiated transform
class UCSDPed2Dataset(Dataset):
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
                    self.paths.append(p)
                    if phase == 'testing' and gt_list is not None:
                        fid = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(re.sub(r'[^0-9]', '', vid)) - 1
                        self.labels.append(1 if fid in gt_list[vid_idx] else 0)
                    else:
                        self.labels.append(0)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase == 'testing' else x

# 2. Load ground truth from .m file
def load_ucsd_gt(root):
    m_files = glob.glob(os.path.join(root, 'Test', '*.m'))
    if not m_files:
        raise FileNotFoundError(f"No .m GT files in {os.path.join(root,'Test')}")
    text = open(m_files[0], 'r').read()
    entries = re.findall(r"TestVideoFile\{end\+1\}\.gt_frame\s*=\s*\[([^\]]+)\];", text)
    if not entries:
        raise ValueError("No gt_frame entries found.")
    gt_list = []
    for entry in entries:
        frames = []
        for part in entry.split(','):
            part = part.strip()
            if ':' in part:
                s,e = map(int, part.split(':'))
                frames.extend(range(s, e+1))
            elif part.isdigit():
                frames.append(int(part))
        gt_list.append(frames)
    return gt_list

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
    def forward(self,x): return self.net(x)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM,128*16*16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Tanh()
        )
    def forward(self,z):
        x = self.fc(z).view(-1,128,16,16)
        return self.net(x)
class DiscriminatorZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM,256), nn.LeakyReLU(0.2),
            nn.Linear(256,1), nn.Sigmoid()
        )
    def forward(self,z): return self.net(z)
class DiscriminatorX(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128*16*16,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

# 4. Training
def train_ocgan_ped2(root, epochs=20, batch_size=32, lr=2e-4):
    gt_list = load_ucsd_gt(root)
    train_tf = transforms.Compose([
        FDATransform(root, patch_ratio=0.1, probability=0.5),
        default_transform
    ])
    ds = UCSDPed2Dataset(root, phase='training', transform=train_tf, gt_list=gt_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    E,G = Encoder().to(device), Generator().to(device)
    D_z,D_x = DiscriminatorZ().to(device), DiscriminatorX().to(device)
    opt_E,opt_G = optim.Adam(E.parameters(),lr=lr), optim.Adam(G.parameters(),lr=lr)
    opt_Dz,opt_Dx = optim.Adam(D_z.parameters(),lr=lr), optim.Adam(D_x.parameters(),lr=lr)
    bce, mse = nn.BCELoss(), nn.MSELoss()
    for ep in range(1,epochs+1):
        for x in loader:
            x = x.to(device)
            b = x.size(0)
            z_real = E(x)
            z_prior = torch.randn(b,LATENT_DIM,device=device)
            x_rec = G(z_real)
            opt_Dz.zero_grad()
            Dz_r, Dz_f = D_z(z_prior), D_z(z_real.detach())
            loss_Dz = bce(Dz_r, torch.ones_like(Dz_r)) + bce(Dz_f, torch.zeros_like(Dz_f))
            loss_Dz.backward(); opt_Dz.step()
            opt_Dx.zero_grad()
            Dx_r, Dx_f = D_x(x), D_x(x_rec.detach())
            loss_Dx = bce(Dx_r, torch.ones_like(Dx_r)) + bce(Dx_f, torch.zeros_like(Dx_f))
            loss_Dx.backward(); opt_Dx.step()
            opt_E.zero_grad(); opt_G.zero_grad()
            loss_Ez = bce(D_z(z_real), torch.ones_like(Dz_r))
            loss_Gx = bce(D_x(x_rec), torch.ones_like(Dx_r))
            loss_rec = mse(x_rec, x)
            loss_total = loss_rec + 0.1*(loss_Ez + loss_Gx)
            loss_total.backward(); opt_E.step(); opt_G.step()
        print(f"Epoch {ep}/{epochs}: D_z={loss_Dz.item():.4f}, D_x={loss_Dx.item():.4f}, EG={loss_total.item():.4f}")
    torch.save(E.state_dict(),'ped2_ocgan_E.pth')
    torch.save(G.state_dict(),'ped2_ocgan_G.pth')
    return E, G, gt_list

# 5. Evaluation
def evaluate_ocgan_ped2(E, G, root, gt_list, batch_size=32):
    E.eval(); G.eval()
    ds = UCSDPed2Dataset(root, phase='testing', transform=default_transform, gt_list=gt_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            z = E(x)
            x_rec = G(z)
            err = torch.mean((x_rec - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist()); labels.extend(y)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, ths = roc_curve(labels, scores)
    thr = ths[np.argmax(tpr-fpr)]
    preds = [1 if s>=thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nConfusion Matrix:\n{cm}\n")
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))
    return scores, labels

def main(dataset_path):
    root = dataset_path
    E,G,gt = train_ocgan_ped2(root, epochs=20, batch_size=32, lr=2e-4)
    evaluate_ocgan_ped2(E,G,root,gt, batch_size=32)

    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
