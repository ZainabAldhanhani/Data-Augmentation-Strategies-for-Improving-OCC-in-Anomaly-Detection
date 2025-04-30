
import argparse
import os
import glob
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_msssim import ssim  # pip install pytorch-msssim
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset loader for UCSD Ped2
default_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class UCSDPed2Dataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or default_transform
        subdir = 'Train' if phase == 'training' else 'Test'
        base = os.path.join(root, subdir)
        vids = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
        self.paths = []
        self.labels = []
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    self.paths.append(p)
                    if phase == 'testing' and gt_list is not None:
                        frame_idx = int(os.path.splitext(os.path.basename(p))[0])
                        # vid is like 'Test001' or 'Train001'
                        vid_idx = int(re.sub('[^0-9]', '', vid)) - 1
                        self.labels.append(1 if frame_idx in gt_list[vid_idx] else 0)
                    else:
                        self.labels.append(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase == 'testing' else x

# 2. Load ground truth from .m file in root or Test folder
import glob, os, re

def load_ucsd_gt(root):
    # look for the .m file in the Test/ folder
    m_files = glob.glob(os.path.join(root, 'Test', '*.m'))
    if not m_files:
        raise FileNotFoundError(f"No .m files found in {os.path.join(root,'Test')} for GT.")
    text = open(m_files[0], 'r').read()

    # match lines like: TestVideoFile{end+1}.gt_frame = [61:180];
    matches = re.findall(
        r"TestVideoFile\{end\+1\}\.gt_frame\s*=\s*\[(\d+):(\d+)\];",
        text
    )
    if not matches:
        raise ValueError("No gt_frame definitions found in TestVideoFile .m file.")

    gt_list = []
    for start_str, end_str in matches:
        start, end = int(start_str), int(end_str)
        # MATLAB ranges are inclusive
        gt_list.append(list(range(start, end+1)))

    return gt_list

# 3. Combined Loss (MSE + L1 + MS-SSIM)
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, recon, x):
        loss_mse = self.mse(recon, x)
        loss_l1 = self.l1(recon, x)
        x01 = (x + 1) / 2
        r01 = (recon + 1) / 2
        ssim_val = ssim(r01, x01, data_range=1.0, size_average=True)
        loss_ssim = 1 - ssim_val
        return self.alpha * loss_mse + self.beta * loss_l1 + self.gamma * loss_ssim

# 4. U-Net Autoencoder
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1,32,3,1,1), nn.ReLU(True))
        self.pool = nn.MaxPool2d(2,2)
        self.enc2 = nn.Sequential(nn.Conv2d(32,64,3,1,1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(64,128,3,1,1), nn.ReLU(True))
        self.up23 = nn.ConvTranspose2d(128,64,2,2)
        self.dec3 = nn.Sequential(nn.Conv2d(128,64,3,1,1), nn.ReLU(True))
        self.up12 = nn.ConvTranspose2d(64,32,2,2)
        self.dec2 = nn.Sequential(nn.Conv2d(64,32,3,1,1), nn.ReLU(True))
        self.final = nn.Conv2d(32,1,1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up23(e3)
        d3 = torch.cat([d3,e2],1)
        d3 = self.dec3(d3)
        d2 = self.up12(d3)
        d2 = torch.cat([d2,e1],1)
        d2 = self.dec2(d2)
        return torch.tanh(self.final(d2))

# 5. Training loop
def train_model(model, loader, epochs=50, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = CombinedLoss()
    for ep in range(1,epochs+1):
        model.train(); total_loss=0
        for batch in loader:
            x = batch[0] if isinstance(batch,(list,tuple)) else batch
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon,x)
            loss.backward(); optimizer.step()
            total_loss += loss.item()*x.size(0)
        avg_loss = total_loss/len(loader.dataset)
        scheduler.step(avg_loss)
        print(f"Epoch {ep}/{epochs}, Loss={avg_loss:.6f}")
    return model

# 6. Evaluation
def evaluate(model,root,gt_list,bs=32):
    model.eval()
    ds = UCSDPed2Dataset(root, phase='testing', gt_list=gt_list)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon-x)**2,[1,2,3]).cpu().numpy()
            scores.extend(err.tolist()); labels.extend(y)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores); thr = th[np.argmax(tpr-fpr)]
    preds = [1 if s>=thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds); acc = accuracy_score(labels, preds); f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f},Acc={acc:.4f},F1={f1:.4f}\nCM:\n{cm}")
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))



def main(dataset_path):
    root = dataset_path
    gt_list = load_ucsd_gt(root)
    train_ds = UCSDPed2Dataset(root, phase='training', gt_list=gt_list)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = UNetAutoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'ucsdped2_unet.pth')
    evaluate(model, root, gt_list)
    
if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
