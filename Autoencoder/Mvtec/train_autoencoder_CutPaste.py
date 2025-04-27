
import argparse
import os
import glob
import random
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
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
        dx = ndi.gaussian_filter((np.random.rand(*arr.shape) * 2 - 1), self.sigma) * self.alpha
        dy = ndi.gaussian_filter((np.random.rand(*arr.shape) * 2 - 1), self.sigma) * self.alpha
        x_grid, y_grid = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
        coords = (y_grid + dy).reshape(-1), (x_grid + dx).reshape(-1)
        distorted = ndi.map_coordinates(arr, coords, order=1, mode='reflect').reshape(arr.shape)
        return Image.fromarray(distorted.astype(np.uint8))

# MixUp & CutMix helpers
def mixup_data(x, alpha=0.4):
    if alpha <= 0:
        return x
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    return lam * x + (1 - lam) * x[index]


def cutmix_data(x, alpha=0.4):
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

# Default test-time transform
def test_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Dataset loader for MVTec AD 
class MVTecDataset(Dataset):
    def __init__(self, root, phase='train', transform=None):
        self.phase = phase
        self.transform = transform
        self.images, self.labels = [], []
        if phase == 'train':
            good_dir = os.path.join(root, 'train', 'good')
            for img_path in sorted(glob.glob(os.path.join(good_dir, '*.png'))):
                self.images.append(img_path)
                self.labels.append(0)
        else:
            test_root = os.path.join(root, 'test')
            for cls in sorted(os.listdir(test_root)):
                cls_dir = os.path.join(test_root, cls)
                if not os.path.isdir(cls_dir):
                    continue
                lbl = 0 if cls == 'good' else 1
                for img_path in sorted(glob.glob(os.path.join(cls_dir, '*.png'))):
                    self.images.append(img_path)
                    self.labels.append(lbl)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        x = self.transform(img)
        return (x, self.labels[idx]) if self.phase != 'train' else x

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
        x01, r01 = (x + 1) / 2, (recon + 1) / 2
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

# Training loop with MixUp & CutMix
def train_model(model, loader, epochs=50, lr=1e-3, mixup_alpha=0.4, cutmix_alpha=0.4, mix_prob=0.5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = CombinedLoss()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            x = x.to(device)
            if random.random() < mix_prob:
                x = mixup_data(x, mixup_alpha) if random.random() < 0.5 else cutmix_data(x, cutmix_alpha)
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
def evaluate(model, root, bs=32):
    model.eval()
    ds = MVTecDataset(root, phase='test', transform=test_transform())
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon - x)**2, dim=[1,2,3]).detach().cpu().numpy()
            scores.extend(err.tolist())
            labels.extend(y)
    scores = np.array(scores)
    labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores)
    thr = th[np.argmax(tpr - fpr)]
    preds = (scores >= thr).astype(int)
    cm = confusion_matrix(labels, preds)
    print(f"Image-level AUC = {auc:.4f}\nConfusion Matrix:\n{cm}")
    print(classification_report(labels, preds, target_names=['Good', 'Defective']))
    return scores, labels



def main(dataset_path):
    root = dataset_path
    # Prepare transforms
    train_transform = transforms.Compose([
        ElasticTransform(alpha=34, sigma=4, probability=0.5),
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_tf = test_transform()

    # Training
    train_ds = MVTecDataset(root, phase='train', transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = conv_autoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'autoencoder_mixcut_elastic.pth')

    # Evaluation
    evaluate(model, root, bs=32)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
