
import argparse
import os
import glob
import re
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_msssim import ssim  # pip install pytorch-msssim
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report
import random
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fourier Domain Adaptation (FDA) augmentation
class FDATransform:
    def __init__(self, root, patch_ratio=0.1, probability=0.5):
        self.probability = probability
        self.patch_ratio = patch_ratio
        base = os.path.join(root, 'Train')
        self.img_paths = []
        vids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                self.img_paths += glob.glob(os.path.join(frame_dir, ext))

    def __call__(self, img):
        if random.random() > self.probability or not self.img_paths:
            return img
        tgt = np.array(img).astype(np.float32)
        src_path = random.choice(self.img_paths)
        src = Image.open(src_path).convert('L').resize(img.size)
        src = np.array(src).astype(np.float32)
        fft_tgt = np.fft.fft2(tgt)
        fft_src = np.fft.fft2(src)
        amp_tgt, pha_tgt = np.abs(fft_tgt), np.angle(fft_tgt)
        amp_src = np.abs(fft_src)
        h, w = tgt.shape
        b = int(min(h, w) * self.patch_ratio)
        cy, cx = h // 2, w // 2
        amp_tgt[cy-b:cy+b, cx-b:cx+b] = amp_src[cy-b:cy+b, cx-b:cx+b]
        fft_new = amp_tgt * np.exp(1j * pha_tgt)
        img_back = np.fft.ifft2(fft_new)
        img_back = np.real(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        return Image.fromarray(img_back)

# Transforms
# NOTE: define `root` before using FDATransform
train_transform = transforms.Compose([
    FDATransform(root, patch_ratio=0.1, probability=0.5),
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Default transform (for testing) (for testing)
def default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Dataset loader for UCSD Ped1
class UCSDPed1Dataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or (train_transform if phase=='training' else test_transform)
        subdir = 'Train' if phase == 'training' else 'Test'
        base_dir = os.path.join(root, subdir)
        vids = sorted(d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)))
        paths, labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base_dir, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    if phase=='testing' and gt_list is not None:
                        idx = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(re.sub(r'[^0-9]', '', vid)) - 1
                        lbl = 1 if idx in gt_list[vid_idx] else 0
                    else:
                        lbl = 0
                    # filter unreadable
                    try:
                        img = Image.open(p).convert('L'); img.close()
                    except:
                        continue
                    paths.append(p)
                    labels.append(lbl)
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase=='testing' else x

# Load ground truth from .m file
def load_ucsd_gt(root):
    test_dir = os.path.join(root, 'Test')
    m_files = glob.glob(os.path.join(test_dir, '*.m'))
    if not m_files:
        raise FileNotFoundError(f"No .m GT in {test_dir}")
    text = open(m_files[0],'r').read()
    entries = re.findall(r"TestVideoFile\{end\+1\}\.gt_frame\s*=\s*\[([^\]]+)\];", text)
    gt_list=[]
    for entry in entries:
        frames=[]
        for part in entry.split(','):
            part=part.strip()
            if ':' in part:
                s,e=map(int,part.split(':'))
                frames.extend(range(s,e+1))
            elif part.isdigit():
                frames.append(int(part))
        gt_list.append(frames)
    return gt_list

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.mse=nn.MSELoss(); self.l1=nn.L1Loss()
        self.alpha,self.beta,self.gamma=alpha,beta,gamma
    def forward(self, recon, x):
        m=self.mse(recon,x); l=self.l1(recon,x)
        x01=(x+1)/2; r01=(recon+1)/2
        s=ssim(r01,x01,data_range=1.0,size_average=True)
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

# Training loop
def train_model(model, loader, epochs=50, lr=1e-3):
    model.to(device)
    opt=optim.Adam(model.parameters(),lr=lr)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,'min',factor=0.5,patience=5)
    crit=CombinedLoss()
    for ep in range(1,epochs+1):
        model.train(); tot=0
        for batch in loader:
            x=batch[0].to(device) if isinstance(batch,(list,tuple)) else batch.to(device)
            opt.zero_grad()
            recon=model(x)
            loss=crit(recon,x)
            loss.backward(); opt.step(); tot+=loss.item()*x.size(0)
        avg=tot/len(loader.dataset); sched.step(avg)
        print(f"Epoch {ep}/{epochs}, Loss={avg:.6f}")
    return model

# Evaluation
def evaluate(model, root, gt_list, bs=32):
    model.eval(); scores,labels=[],[]
    ds=UCSDPed1Dataset(root,'testing',transform=test_transform,gt_list=gt_list)
    loader=DataLoader(ds,batch_size=bs,shuffle=False)
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); r=model(x)
            err=torch.mean((r-x)**2,dim=[1,2,3]).detach().cpu().numpy()
            scores+=err.tolist(); labels+=y
    auc=roc_auc_score(labels,scores)
    fpr,tpr,th=roc_curve(labels,scores);thr=th[np.argmax(tpr-fpr)]
    preds=[1 if s>=thr else 0 for s in scores]
    cm=confusion_matrix(labels,preds)
    pr_auc = average_precision_score(labels, preds)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={accuracy_score(labels,preds):.4f}, F1={f1_score(labels,preds):.4f}\nCM:\n{cm}")
    print(classification_report(labels,preds,target_names=['Normal','Anomaly']))

 
def main(dataset_path):
    root = dataset_path
    gt_list=load_ucsd_gt(root)
    train_ds=UCSDPed1Dataset(root,'training',transform=train_transform,gt_list=gt_list)
    train_loader=DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=4)
    model=conv_autoencoder()
    model=train_model(model,train_loader,epochs=50,lr=1e-3)
    torch.save(model.state_dict(),'ucsdped1_env.pth')
    evaluate(model,root,gt_list,bs=32)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
