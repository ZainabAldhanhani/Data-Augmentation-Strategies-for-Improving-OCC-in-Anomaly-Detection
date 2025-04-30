import argparse
import os
import glob
import scipy.io as sio
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

# 1. Dataset loader for Avenue
class AvenueDataset(Dataset):
    def __init__(self, root, phase='train', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        base = os.path.join(root, phase, 'frames')
        vids = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
        self.paths, self.labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    self.paths.append(p)
                    if phase=='testing':
                        fid = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(vid)-1
                        self.labels.append(1 if fid in gt_list[vid_idx] else 0)
                    else:
                        self.labels.append(0)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase=='testing' else x

# 2. Load ground truth
def load_avenue_gt(root):
    mat = sio.loadmat(os.path.join(root,'avenue.mat'))
    gt_cell = mat.get('gt', mat.get('gt_frame', None))[0]
    return [list(map(int, arr.flatten())) for arr in gt_cell]

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

# 4. OCGAN training loop

def train_ocgan(root, epochs=50, batch_size=32, lr=2e-4):
    # prepare data
    gt_list = load_avenue_gt(root)
    ds = AvenueDataset(root,'training')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    # init models
    E = Encoder().to(device)
    G = Generator().to(device)
    D_z = DiscriminatorZ().to(device)
    D_x = DiscriminatorX().to(device)
    # optimizers
    opt_E = optim.Adam(E.parameters(),lr=lr)
    opt_G = optim.Adam(G.parameters(),lr=lr)
    opt_Dz = optim.Adam(D_z.parameters(),lr=lr)
    opt_Dx = optim.Adam(D_x.parameters(),lr=lr)
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    real_label, fake_label = 1., 0.

    for ep in range(1,epochs+1):
        for x in loader:
            x = x.to(device)
            batch = x.size(0)
            # encode real latent
            z_real = E(x)
            # sample prior z
            z_prior = torch.randn(batch,LATENT_DIM,device=device)
            # generator output
            x_rec = G(z_real)
            # ========== update D_z ==========
            opt_Dz.zero_grad()
            Dz_real = D_z(z_prior)
            Dz_fake = D_z(z_real.detach())
            loss_Dz = bce(Dz_real, torch.full_like(Dz_real, real_label)) + bce(Dz_fake, torch.full_like(Dz_fake, fake_label))
            loss_Dz.backward(); opt_Dz.step()
            # ========== update D_x ==========
            opt_Dx.zero_grad()
            Dx_real = D_x(x)
            Dx_fake = D_x(x_rec.detach())
            loss_Dx = bce(Dx_real, torch.full_like(Dx_real, real_label)) + bce(Dx_fake, torch.full_like(Dx_fake, fake_label))
            loss_Dx.backward(); opt_Dx.step()
            # ========== update E, G ==========
            opt_E.zero_grad(); opt_G.zero_grad()
            # adversarial loss on z
            Dz_fake2 = D_z(z_real)
            loss_Ez = bce(Dz_fake2, torch.full_like(Dz_fake2, real_label))
            # adversarial loss on x
            Dx_fake2 = D_x(x_rec)
            loss_Gx = bce(Dx_fake2, torch.full_like(Dx_fake2, real_label))
            # reconstruction loss
            loss_rec = mse(x_rec, x)
            loss_total = loss_rec + 0.1*(loss_Ez + loss_Gx)
            loss_total.backward(); opt_E.step(); opt_G.step()
        print(f"Epoch {ep}/{epochs}, Dz={loss_Dz.item():.4f}, Dx={loss_Dx.item():.4f}, E+G={loss_total.item():.4f}")
    # save models
    torch.save(E.state_dict(),'avenue_ocgan_E.pth')
    torch.save(G.state_dict(),'avenue_ocgan_G.pth')
    return E, G, gt_list

# 5. Evaluation using reconstruction error


def evaluate_ocgan(E, G, root, gt_list, batch_size=32):
    E.eval(); G.eval()
    ds = AvenueDataset(root,'testing',gt_list=gt_list)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            z = E(x)
            x_rec = G(z)
            err = torch.mean((x_rec - x)**2,dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist()); labels.extend(y)
    auc = roc_auc_score(labels,scores)
    fpr,tpr,ths = roc_curve(labels,scores)
    thr = ths[np.argmax(tpr-fpr)]
    preds = [1 if s>=thr else 0 for s in scores]
    cm = confusion_matrix(labels,preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nConfusion Matrix:\n{cm}\n")
    #print(f"Frame-level AUC={auc:.4f}\nConfusion Matrix:\n{cm}\n")
    print(classification_report(labels,preds,target_names=['Normal','Anomaly']))
    return scores,labels

def main(dataset_path):
    root = dataset_path
    E,G,gt = train_ocgan(root,epochs=50,batch_size=32,lr=2e-4)
    evaluate_ocgan(E,G,root,gt,batch_size=32)


if __name__ == '__main__':
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
