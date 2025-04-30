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

# Simulated Environmental Effects augmentation
class EnvironmentalTransform:
    def __init__(self, rain_prob=0.3, fog_prob=0.3, sun_prob=0.3):
        self.rain_prob = rain_prob
        self.fog_prob = fog_prob
        self.sun_prob = sun_prob

    def __call__(self, img):
        # Rain effect
        if random.random() < self.rain_prob:
            img = self._add_rain(img)
        # Fog effect
        if random.random() < self.fog_prob:
            img = self._add_fog(img)
        # Sun glare
        if random.random() < self.sun_prob:
            img = self._add_sun_glare(img)
        return img

    def _add_rain(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        h, w = arr.shape
        rain = np.zeros((h, w), dtype=np.uint8)
        drops = int(h * w * 0.0005)
        for _ in range(drops):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            length = random.randint(10, 20)
            for i in range(length):
                yy = min(h-1, y+i)
                rain[yy, x] = 255
        rain_img = Image.fromarray(rain).filter(ImageFilter.GaussianBlur(1))
        return Image.blend(img, rain_img, alpha=0.3)

    def _add_fog(self, img: Image.Image) -> Image.Image:
        fog = Image.new('L', img.size, color=255)
        fog = fog.filter(ImageFilter.GaussianBlur(radius=img.size[0]//15))
        return Image.blend(img, fog, alpha=0.4)

    def _add_sun_glare(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        mask = Image.new('L', (w, h), 0)
        cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        rad = random.randint(min(w,h)//8, min(w,h)//4)
        yy, xx = np.ogrid[:h, :w]
        circle = ((xx-cx)**2 + (yy-cy)**2) <= rad**2
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        mask_arr[circle] = 255
        mask = Image.fromarray(mask_arr)
        bright = ImageEnhance.Brightness(img).enhance(1.5)
        return Image.composite(bright, img, mask)

# Transforms
train_transform = transforms.Compose([
    EnvironmentalTransform(rain_prob=0.3, fog_prob=0.3, sun_prob=0.3),
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

# Default transform (for testing)
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
