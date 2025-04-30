import os
import glob
import random
import math
import scipy.io as sio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from pytorch_msssim import ssim  # pip install pytorch-msssim
import argparse
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CutPaste augmentation: randomly cut a square patch and paste at a new location
class CutPaste:
    def __init__(self, patch_area_ratio=0.1, probability=0.5):
        self.patch_area_ratio = patch_area_ratio
        self.probability = probability

    def __call__(self, img):
        # img: PIL.Image in grayscale
        if random.random() > self.probability:
            return img
        w, h = img.size
        patch_size = int(math.sqrt(self.patch_area_ratio * w * h))
        # ensure patch fits
        if patch_size < 1:
            return img
        x1 = random.randint(0, w - patch_size)
        y1 = random.randint(0, h - patch_size)
        patch = img.crop((x1, y1, x1 + patch_size, y1 + patch_size))
        x2 = random.randint(0, w - patch_size)
        y2 = random.randint(0, h - patch_size)
        img_copy = img.copy()
        img_copy.paste(patch, (x2, y2))
        return img_copy

# Default transforms: apply augmentation in training
default_transform_train = transforms.Compose([
    CutPaste(patch_area_ratio=0.1, probability=0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

default_transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 1. Dataset loader for Avenue
class AvenueDataset(Dataset):
    def __init__(self, root, phase='train', transform=None, gt_list=None):
        self.phase = phase
        if transform is None:
            self.transform = default_transform_train if phase == 'training' else default_transform_test
        else:
            self.transform = transform
        base = os.path.join(root, phase, 'frames')
        vids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        self.paths, self.labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    self.paths.append(p)
                    if phase == 'testing' and gt_list is not None:
                        frame_idx = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(vid) - 1
                        self.labels.append(1 if frame_idx in gt_list[vid_idx] else 0)
                    else:
                        self.labels.append(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('L')
        x = self.transform(img)
        if self.phase == 'testing':
            return x, self.labels[idx]
        return x

# 2. Load ground truth
def load_avenue_gt(root):
    mat = sio.loadmat(os.path.join(root, 'avenue.mat'))
    gt_cell = mat.get('gt', mat.get('gt_frame', None))[0]
    return [list(map(int, arr.flatten())) for arr in gt_cell]

# 3. Combined loss (MSE + L1 + MS-SSIM)
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
        x_01 = (x + 1) / 2
        recon_01 = (recon + 1) / 2
        ssim_val = ssim(recon_01, x_01, data_range=1.0, size_average=True)
        loss_ssim = 1 - ssim_val
        return self.alpha * loss_mse + self.beta * loss_l1 + self.gamma * loss_ssim

# 4. ConvNeXt-based autoencoder
def convnext_autoencoder():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    backbone = convnext_tiny(weights=weights)
    encoder = backbone.features
    for param in encoder.parameters():
        param.requires_grad = False
    decoder = nn.Sequential(
        nn.ConvTranspose2d(768, 512, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
    )
    return nn.Sequential(encoder, decoder)

# 5. Training loop
def train_model(model, loader, epochs=50, lr=1e-3):
    model.to(device)
    decoder_params = list(model[1].parameters())
    optimizer = optim.Adam(decoder_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = CombinedLoss()

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()
            recon = model(inputs)
            loss = criterion(recon, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        avg_loss = running_loss / len(loader.dataset)
        scheduler.step(avg_loss)
        print(f"Epoch {ep}/{epochs}, Loss={avg_loss:.6f}")
    return model

# 6. Evaluation
def evaluate(model, root, gt_list, bs=32):
    model.eval()
    ds = AvenueDataset(root, phase='testing', gt_list=gt_list)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon - x) ** 2, dim=[1, 2, 3]).cpu().numpy()
            scores.extend(err.tolist())
            labels.extend(y)
    from sklearn.metrics import roc_auc_score as A, roc_curve, confusion_matrix as C, accuracy_score as Ac, f1_score as F
    auc = A(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores)
    thr = th[np.argmax(tpr - fpr)]
    pred = [1 if s >= thr else 0 for s in scores]
    cm = C(labels, pred)
    acc = Ac(labels, pred)
    f1 = F(labels, pred)
    pr_auc = average_precision_score(labels, pred)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nCM:\n{cm}")

# 7. Run
def main(dataset_path):
    root = dataset_path
    gt_list = load_avenue_gt(root)
    train_ds = AvenueDataset(root, phase='training')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = convnext_autoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'autoencoder_avenue_CutPaste.pth')
    evaluate(model, root, gt_list)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
