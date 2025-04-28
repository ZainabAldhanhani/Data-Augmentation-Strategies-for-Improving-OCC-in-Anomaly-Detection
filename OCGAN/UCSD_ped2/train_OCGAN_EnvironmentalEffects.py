import argparse
import os
import glob
import re
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulated Environmental Effects augmentation
class SimulatedEnvEffects:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        arr = np.array(img).astype(np.float32)
        # add random gaussian noise
        noise = np.random.normal(0, 10, arr.shape)
        arr = arr + noise
        # simulate blur
        if random.random() < 0.5:
            img_blur = Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5))
            )
            arr = np.array(img_blur).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

# Default transform (no augmentation)
def get_default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

default_transform = get_default_transform()

# 1. Load ground truth from .m file (Ped2)
def load_ucsd_gt(root):
    m_files = glob.glob(os.path.join(root, 'Test', '*.m'))
    if not m_files:
        raise FileNotFoundError(f"No .m GT files in {os.path.join(root,'Test')}")
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

# 2. Dataset loader for UCSD Ped2
class UCSDPed2Dataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or default_transform
        subdir = 'Train' if phase=='training' else 'Test'
        base = os.path.join(root, subdir)
        vids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        self.paths, self.labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    self.paths.append(p)
                    if phase=='testing' and gt_list:
                        fid = int(os.path.splitext(os.path.basename(p))[0])
                        idx = int(re.sub(r'[^0-9]', '', vid)) - 1
                        self.labels.append(1 if fid in gt_list[idx] else 0)
                    else:
                        self.labels.append(0)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase=='testing' else x

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
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Tanh()
        )
    def forward(self,z):
        x = self.fc(z).view(-1,128,16,16)
        return self.deconv(x)

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

# 4. Training loop for Ped2
def train_ocgan_ped2(root, epochs=20, batch_size=32, lr=2e-4):
    gt_list = load_ucsd_gt(root)
    train_tf = transforms.Compose([
        SimulatedEnvEffects(probability=0.5),
        default_transform
    ])
    ds = UCSDPed2Dataset(root,'training',transform=train_tf,gt_list=gt_list)
    loader = DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=4)
    E,G = Encoder().to(device), Generator().to(device)
    D_z,D_x = DiscriminatorZ().to(device), DiscriminatorX().to(device)
    optE,optG = optim.Adam(E.parameters(),lr=lr), optim.Adam(G.parameters(),lr=lr)
    optDz,optDx = optim.Adam(D_z.parameters(),lr=lr), optim.Adam(D_x.parameters(),lr=lr)
    bce,mse = nn.BCELoss(), nn.MSELoss()
    for ep in range(1,epochs+1):
        for x in loader:
            x = x.to(device); b=x.size(0)
            z_real = E(x); z_prior=torch.randn(b,LATENT_DIM,device=device)
            x_rec = G(z_real)
            optDz.zero_grad()
            lossDz = bce(D_z(z_prior),torch.ones_like(D_z(z_prior))) + bce(D_z(z_real.detach()),torch.zeros_like(D_z(z_real)))
            lossDz.backward(); optDz.step()
            optDx.zero_grad()
            lossDx = bce(D_x(x),torch.ones_like(D_x(x))) + bce(D_x(x_rec.detach()),torch.zeros_like(D_x(x_rec)))
            lossDx.backward(); optDx.step()
            optE.zero_grad(); optG.zero_grad()
            lossEz = bce(D_z(z_real), torch.ones_like(D_z(z_real)))
            lossGx = bce(D_x(x_rec), torch.ones_like(D_x(x_rec)))
            lossRec= mse(x_rec,x)
            loss= lossRec + 0.1*(lossEz+lossGx)
            loss.backward(); optE.step(); optG.step()
        print(f"Epoch {ep}/{epochs}: Dz={lossDz.item():.4f}, Dx={lossDx.item():.4f}, EG={loss.item():.4f}")
    torch.save(E.state_dict(),'ped2_ocgan_E.pth')
    torch.save(G.state_dict(),'ped2_ocgan_G.pth')
    return E,G,gt_list

# 5. Evaluation for Ped2
def evaluate_ocgan_ped2(E,G,root,gt_list,batch_size=32):
    E.eval(); G.eval()
    ds=UCSDPed2Dataset(root,'testing',transform=default_transform,gt_list=gt_list)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False)
    scores,labels=[],[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            x_rec = G(E(x))
            err = torch.mean((x_rec - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist()); labels.extend(y)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, ths = roc_curve(labels, scores)
    thr = ths[np.argmax(tpr-fpr)]
    preds = [1 if s>=thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nConfusion Matrix:\n{cm}")
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))



def main(dataset_path):
    root = dataset_path
    E,G,gt_list = train_ocgan_ped2(root)
    evaluate_ocgan_ped2(E,G,root,gt_list)

if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
