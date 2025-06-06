import argparse
import os
import glob
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
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

# Simulated Environmental Effects augmentation
class EnvironmentalTransform:
    def __init__(self, rain_prob=0.3, fog_prob=0.3, sun_prob=0.3):
        self.rain_prob = rain_prob
        self.fog_prob = fog_prob
        self.sun_prob = sun_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.rain_prob:
            img = self._add_rain(img)
        if random.random() < self.fog_prob:
            img = self._add_fog(img)
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
        fog = Image.new('L', img.size, 255)
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
        self.transform = transform or test_transform()
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
        m = self.mse(recon, x)
        l = self.l1(recon, x)
        x01, r01 = (x + 1) / 2, (recon + 1) / 2
        s = ssim(r01, x01, data_range=1.0, size_average=True)
        return self.alpha * m + self.beta * l + self.gamma * (1 - s)

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

# Training loop without MixUp/CutMix
def train_model(model, loader, epochs=50, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = CombinedLoss()
    for ep in range(1, epochs + 1):
        model.train(); total_loss = 0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward(); optimizer.step(); total_loss += loss.item() * x.size(0)
        avg = total_loss / len(loader.dataset); scheduler.step(avg)
        print(f"Epoch {ep}/{epochs}, Loss={avg:.6f}")
    return model
    
def evaluate(model, root, bs=32):
    model.eval()
    ds = MVTecBottleDataset(root, phase='test', transform=test_transform)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon-x)**2, dim=[1,2,3]).detach().cpu().numpy()
            scores.extend(err.tolist())
            labels.extend(y)
    scores = np.array(scores); labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, th = roc_curve(labels, scores)
    thr = th[np.argmax(tpr - fpr)]
    preds = (scores >= thr).astype(int)
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    print("PR-AUC Score:", pr_auc)
    print(f"AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}\nConfusion Matrix:\n{cm}")
    #print(f"Image-level AUC = {auc:.4f}\nConfusion Matrix:\n{cm}")
    print(classification_report(labels, preds, target_names=['Good','Defective']))
    return scores, labels

def main(dataset_path):
    root = dataset_path
    train_transform = transforms.Compose([
        EnvironmentalTransform(rain_prob=0.3, fog_prob=0.3, sun_prob=0.3),
        transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    test_tf = test_transform()

    train_ds = MVTecDataset(root, phase='train', transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = conv_autoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'autoencoder_env.pth')

    evaluate(model, root, bs=32)

    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
