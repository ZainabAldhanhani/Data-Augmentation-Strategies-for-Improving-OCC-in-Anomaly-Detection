
import argparse
import os
import glob
import re
import random
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_msssim import ssim  # pip install pytorch-msssim
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Elastic deformation augmentation
class ElasticTransform:
    def __init__(self, alpha=34, sigma=4, probability=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img
        arr = np.array(img)
        shape = arr.shape
        dx = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,)))
        distorted = ndi.map_coordinates(arr, indices, order=1, mode='reflect').reshape(shape)
        return Image.fromarray(distorted.astype(np.uint8))

# Transforms
train_transform = transforms.Compose([
    ElasticTransform(alpha=34, sigma=4, probability=0.5),
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

# Dataset loader for UCSD Ped1
class UCSDPed1Dataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or (train_transform if phase == 'training' else test_transform)
        subdir = 'Train' if phase == 'training' else 'Test'
        base_dir = os.path.join(root, subdir)
        vids = sorted(d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)))
        paths, labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base_dir, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    lbl = 0
                    if phase == 'testing' and gt_list is not None:
                        idx = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(re.sub(r'[^0-9]', '', vid)) - 1
                        lbl = 1 if idx in gt_list[vid_idx] else 0
                    try:
                        img = Image.open(p).convert('L')
                        img.close()
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
        x01, r01 = (x + 1) / 2, (recon + 1) / 2
        s = ssim(r01, x01, data_range=1.0, size_average=True)
        return self.alpha * m + self.beta * l + self.gamma * (1 - s)

# Simplified Autoencoder
def conv_autoencoder():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(True),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
        nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(True),
        nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(True),
        nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Tanh()
    )

# Training loop
def train_model(model, loader, epochs=50, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = CombinedLoss()
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        scheduler.step(avg_loss)
        print(f"Epoch {ep}/{epochs}, Loss={avg_loss:.6f}")
    return model

# Evaluation
def evaluate(model, root, gt_list, bs=32):
    model.eval()
    scores, labels = [], []
    ds = UCSDPed1Dataset(root, 'testing', transform=test_transform, gt_list=gt_list)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon - x)**2, dim=[1,2,3]).detach().cpu().numpy()
            scores.extend(err.tolist())
            labels.extend(y)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores)
    thr = th[np.argmax(tpr - fpr)]
    preds = [1 if s >= thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds)
    print(f"AUC={auc:.4f}, Acc={accuracy_score(labels,preds):.4f}, F1={f1_score(labels,preds):.4f}\nCM:\n{cm}")
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))


def main(dataset_path):
    root = dataset_path
    gt_list = load_ucsd_gt(root)
    train_ds = UCSDPed1Dataset(root, 'training', transform=train_transform, gt_list=gt_list)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = conv_autoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'ucsdped1_elastic.pth')
    evaluate(model, root, gt_list, bs=32)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
