
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

# Default transform (no augmentation)
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
        self.transform = transform or default_transform()
        subdir = 'Train' if phase == 'training' else 'Test'
        base_dir = os.path.join(root, subdir)
        vids = sorted(d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)))
        temp_paths, temp_labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base_dir, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    label = 0
                    if phase == 'testing' and gt_list is not None:
                        frame_idx = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(re.sub(r'[^0-9]', '', vid)) - 1
                        label = 1 if frame_idx in gt_list[vid_idx] else 0
                    # Test full loadability including conversion
                    try:
                        img = Image.open(p)
                        img = img.convert('L')
                        img.close()
                    except Exception:
                        continue
                    temp_paths.append(p)
                    temp_labels.append(label)
        self.paths = temp_paths
        self.labels = temp_labels
        self.labels = temp_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('L')
        except Exception as e:
            print(f"Error loading image at: {path}\nException: {e}")
            raise
        x = self.transform(img)
        if self.phase == 'testing':
            return x, self.labels[idx]
        return x

# Load ground truth from .m file (handles multiple segments per video)
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
                start, end = map(int, part.split(':'))
                frames.extend(list(range(start, end+1)))
            elif part.isdigit():
                frames.append(int(part))
        gt_list.append(frames)
    return gt_list

# Combined Loss (MSE + L1 + MS-SSIM)
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

# Simplified convolutional autoencoder
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
    for ep in range(1, epochs + 1):
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
    ds = UCSDPed1Dataset(root, 'testing', transform=default_transform(), gt_list=gt_list)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
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
    pr_auc = average_precision_score(y_true, y_scores)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={accuracy_score(labels,preds):.4f}, F1={f1_score(labels,preds):.4f}\nCM:\n{cm}")
    print(classification_report(labels, preds, target_names=['Normal','Anomaly']))

# 7. Main
def main(dataset_path):
    root = dataset_path
    gt_list = load_ucsd_gt(root)
    train_ds = UCSDPed1Dataset(root, 'training', transform=default_transform(), gt_list=gt_list)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = conv_autoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'ucsdped1_noaug.pth')
    evaluate(model, root, gt_list, bs=32)
    
if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
