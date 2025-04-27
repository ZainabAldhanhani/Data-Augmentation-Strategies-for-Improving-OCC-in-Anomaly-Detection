
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
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default transform (no augmentation)
def default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# MixUp and CutMix helpers
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
    x_cut = x.clone()
    x_cut[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    return x_cut

# Dataset loader for UCSD Ped1
class UCSDPed1Dataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or default_transform()
        subdir = 'Train' if phase == 'training' else 'Test'
        base_dir = os.path.join(root, subdir)
        vids = sorted(d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)))
        paths, labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base_dir, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    if phase == 'testing' and gt_list is not None:
                        idx = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(re.sub(r'[^0-9]', '', vid)) - 1
                        lbl = 1 if idx in gt_list[vid_idx] else 0
                    else:
                        lbl = 0
                    try:
                        img = Image.open(p).convert('L'); img.close()
                    except:
                        continue
                    paths.append(p)
                    labels.append(lbl)
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        if self.phase == 'testing':
            return x, self.labels[idx]
        return x

# Load ground truth from .m file
def load_ucsd_gt(root):
    test_dir = os.path.join(root, 'Test')
    m_files = glob.glob(os.path.join(test_dir, '*.m'))
    if not m_files:
        raise FileNotFoundError(f"No .m GT in {test_dir}")
    text = open(m_files[0], 'r').read()
    entries = re.findall(r"TestVideoFile\{end\+1\}\.gt_frame\s*=\s*\[([^\]]+)\];", text)
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

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    def forward(self, recon, x):
        m = self.mse(recon, x)
        l = self.l1(recon, x)
        x01, r01 = (x+1)/2, (recon+1)/2
        s = ssim(r01, x01, data_range=1.0, size_average=True)
        return self.alpha*m + self.beta*l + self.gamma*(1-s)

# Simplified Autoencoder
def conv_autoencoder():
    return nn.Sequential(
        nn.Conv2d(1,32,3,2,1), nn.ReLU(True),
        nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
        nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        nn.ConvTranspose2d(128,64,3,2,1,1), nn.ReLU(True),
        nn.ConvTranspose2d(64,32,3,2,1,1), nn.ReLU(True),
        nn.ConvTranspose2d(32,1,3,2,1,1), nn.Tanh()
    )

# Training loop with MixUp & CutMix
def train_model(model, loader, epochs=50, lr=1e-3, mixup_alpha=0.4, cutmix_alpha=0.4, mix_prob=0.5):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
    crit = CombinedLoss()
    for ep in range(1, epochs+1):
        model.train(); total=0
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch,(list,tuple)) else batch.to(device)
            if random.random() < mix_prob:
                x = mixup_data(x, mixup_alpha) if random.random()<0.5 else cutmix_data(x, cutmix_alpha)
            opt.zero_grad()
            recon = model(x)
            loss = crit(recon, x)
            loss.backward(); opt.step(); total += loss.item()*x.size(0)
        avg = total/len(loader.dataset); sched.step(avg)
        print(f"Epoch {ep}/{epochs}, Loss={avg:.6f}")
    return model

# Evaluation
def evaluate(model, root, gt_list, bs=32):
    model.eval(); scores,labels=[],[]
    ds = UCSDPed1Dataset(root,'testing',transform=default_transform(),gt_list=gt_list)
    for x,y in DataLoader(ds,bs,False):
        x=x.to(device); r=model(x)
        err = torch.mean((r - x)**2, dim=[1,2,3]).detach().cpu().numpy()
        scores+=err.tolist(); labels+=y
    auc=roc_auc_score(labels,scores)
    fpr,tpr,th=roc_curve(labels,scores);thr=th[np.argmax(tpr-fpr)]
    preds=[1 if s>=thr else 0 for s in scores]
    cm=confusion_matrix(labels,preds)
    print(f"AUC={auc:.4f}, Acc={accuracy_score(labels,preds):.4f}, F1={f1_score(labels,preds):.4f}\nCM:\n{cm}")
    print(classification_report(labels,preds,target_names=['Normal','Anomaly']))


def main(dataset_path):
    root = dataset_path
    gt_list=load_ucsd_gt(root)
    train_ds=UCSDPed1Dataset(root,'training',transform=default_transform(),gt_list=gt_list)
    train_loader=DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=4)
    model=conv_autoencoder()
    model=train_model(model,train_loader,epochs=50,lr=1e-3)
    torch.save(model.state_dict(),'ucsdped1_mix.pth')
    evaluate(model,root,gt_list,bs=32)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
