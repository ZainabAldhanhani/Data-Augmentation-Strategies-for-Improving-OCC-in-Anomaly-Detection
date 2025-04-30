import os
import glob
import random
import math
import scipy.io as sio
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from pytorch_msssim import ssim  # pip install pytorch-msssim
from sklearn.metrics import roc_auc_score as A, roc_curve, confusion_matrix as C, accuracy_score as Ac, f1_score as F
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import argparse
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Elastic deformation augmentation
class ElasticTransform:
    def __init__(self, alpha=34, sigma=4, probability=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability

    def __call__(self, img):
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
default_transform_train = transforms.Compose([
    ElasticTransform(alpha=34, sigma=4, probability=0.5),
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

# Dataset loader for Avenue
default_phase_map = {'training': default_transform_train, 'testing': default_transform_test}
class AvenueDataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.transform = transform or default_phase_map[phase]
        base = os.path.join(root, phase, 'frames')
        vids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        self.paths, self.labels = [], []
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                for p in sorted(glob.glob(os.path.join(frame_dir, ext))):
                    self.paths.append(p)
                    if phase == 'testing' and gt_list is not None:
                        idx = int(os.path.splitext(os.path.basename(p))[0])
                        vid_idx = int(vid) - 1
                        self.labels.append(1 if idx in gt_list[vid_idx] else 0)
                    else:
                        self.labels.append(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase == 'testing' else x

# Load ground truth
def load_avenue_gt(root):
    mat = sio.loadmat(os.path.join(root, 'avenue.mat'))
    gt = mat.get('gt', mat.get('gt_frame', None))[0]
    return [list(map(int, arr.flatten())) for arr in gt]

# Combined loss (MSE + L1 + MS-SSIM)
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

# ConvNeXt-based autoencoder
def convnext_autoencoder():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    encoder = model.features
    for p in encoder.parameters(): p.requires_grad = False
    decoder = nn.Sequential(
        nn.ConvTranspose2d(768, 512, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
    )
    return nn.Sequential(encoder, decoder)

# Training loop
def train_model(model, loader, epochs=50, lr=1e-3):
    model.to(device)
    params = model[1].parameters()
    opt = optim.Adam(params, lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
    crit = CombinedLoss()
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = crit(recon, x)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(loader.dataset)
        sched.step(avg)
        print(f"Epoch {ep}/{epochs}, Loss={avg:.6f}")
    return model

# Evaluation
def evaluate(model, root, gt_list, bs=32):
    model.eval()
    ds = AvenueDataset(root, phase='testing', gt_list=gt_list)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon - x)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(err.tolist())
            labels.extend(y)
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report
    auc = roc_auc_score(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores)
    thr = th[np.argmax(tpr - fpr)]
    preds = [1 if s>=thr else 0 for s in scores]
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, pred)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nConfusion Matrix:\n{cm}")
    print("Classification Report:\n", classification_report(labels, preds, target_names=['Normal','Anomaly']))


# 7. Run
def main(dataset_path):
    root = dataset_path
    gt = load_avenue_gt(root)
    train_ds = AvenueDataset(root, phase='training', gt_list=gt)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = convnext_autoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'convnext_avenue.pth')
    evaluate(model, root, gt)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
