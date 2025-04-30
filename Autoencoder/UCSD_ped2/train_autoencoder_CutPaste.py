
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
from pytorch_msssim import ssim  # pip install pytorch-msssim
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0. MixUp and CutMix helper functions
def mixup_data(x, alpha=1.0):
    if alpha <= 0:
        return x
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    return lam * x + (1 - lam) * x[index]


def cutmix_data(x, alpha=1.0):
    if alpha <= 0:
        return x
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    x_cutmix = x.clone()
    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    return x_cutmix

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
        base_dir = os.path.join(root, subdir)
        vids = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        self.paths = []
        self.labels = []
        for vid in vids:
            frame_dir = os.path.join(base_dir, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    self.paths.append(p)
                    if phase == 'testing' and gt_list is not None:
                        frame_idx = int(os.path.splitext(os.path.basename(p))[0])
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

# 2. Load ground truth from .m file in Test folder
def load_ucsd_gt(root):
    test_dir = os.path.join(root, 'Test')
    m_files = glob.glob(os.path.join(test_dir, '*.m'))
    if not m_files:
        raise FileNotFoundError(f"No .m files found in {test_dir} for GT.")
    text = open(m_files[0], 'r').read()
    # parse lines: TestVideoFile{end+1}.gt_frame = [start:end];
    matches = re.findall(r"TestVideoFile\{end\+1\}\.gt_frame\s*=\s*\[(\d+):(\d+)\];", text)
    if not matches:
        raise ValueError("No gt_frame definitions in TestVideoFile .m file.")
    gt_list = []
    for s_str, e_str in matches:
        s, e = int(s_str), int(e_str)
        gt_list.append(list(range(s, e+1)))
    return gt_list

# 3. Combined Loss (MSE + L1 + MS-SSIM)
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, recon, x):
        loss_mse = self.mse(recon, x)
        loss_l1 = self.l1(recon, x)
        x01 = (x + 1) / 2
        r01 = (recon + 1) / 2
        ssim_val = ssim(r01, x01, data_range=1.0, size_average=True)
        loss_ssim = 1 - ssim_val
        return self.alpha * loss_mse + self.beta * loss_l1 + self.gamma * loss_ssim

# 4. Simplified Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(True),  # 128->64
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),  # 64->32
            nn.Conv2d(64,128, 3, 2, 1), nn.ReLU(True)   # 32->16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1,1), nn.ReLU(True),  # 16->32
            nn.ConvTranspose2d(64,32,3,2,1,1), nn.ReLU(True),   # 32->64
            nn.ConvTranspose2d(32, 1,3,2,1,1), nn.Tanh()        # 64->128
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# 5. Trainer with MixUp & CutMix
class Trainer:
    def __init__(self, model, loader, lr=1e-3):
        self.model = model.to(device)
        self.loader = loader
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', factor=0.5, patience=5)
        self.crit = CombinedLoss()

    def train(self, epochs=50, mixup_alpha=0.4, cutmix_alpha=0.4, mix_prob=0.5):
        for ep in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in self.loader:
                # handle either (x, labels) or x-only batches
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                # apply MixUp or CutMix randomly
                if random.random() < mix_prob:
                    if random.random() < 0.5:
                        x = mixup_data(x, mixup_alpha)
                    else:
                        x = cutmix_data(x, cutmix_alpha)
                self.opt.zero_grad()
                recon = self.model(x)
                loss = self.crit(recon, x)
                loss.backward()
                self.opt.step()
                running_loss += loss.item() * x.size(0)
            avg_loss = running_loss / len(self.loader.dataset)
            self.sched.step(avg_loss)
            print(f"Epoch {ep}/{epochs}, Loss={avg_loss:.6f}")
        return self.model

# 6. Evaluation Evaluation
def evaluate(model, root, gt_list, bs=32):
    model.eval()
    ds = UCSDPed2Dataset(root, phase='testing', gt_list=gt_list)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist())
            labels.extend(y)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores)
    thr = th[np.argmax(tpr - fpr)]
    preds = [1 if s >= thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    print("PR-AUC Score:", pr_auc)

    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nCM:\n{cm}")
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))



def main(dataset_path):
    root = dataset_path
    gt_list = load_ucsd_gt(root)
    train_ds = UCSDPed2Dataset(root, phase='training', gt_list=gt_list)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = Autoencoder()
    trainer = Trainer(model, train_loader)
    model = trainer.train(epochs=50, mixup_alpha=0.4, cutmix_alpha=0.4, mix_prob=0.5)
    torch.save(model.state_dict(), 'ucsdped2_auto_mix.pth')
    evaluate(model, root, gt_list)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
