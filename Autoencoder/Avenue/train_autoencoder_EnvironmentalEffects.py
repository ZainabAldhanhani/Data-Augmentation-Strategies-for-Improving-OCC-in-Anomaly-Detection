import argparse
import os
import glob
import random
import scipy.io as sio
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
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
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulated Environmental Effects augmentation
class EnvironmentalTransform:
    def __init__(self, rain_prob=0.3, fog_prob=0.3, sun_prob=0.3):
        self.rain_prob = rain_prob
        self.fog_prob = fog_prob
        self.sun_prob = sun_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        # Rain effect: overlay streaks
        if random.random() < self.rain_prob:
            img = self._add_rain(img)
        # Fog effect: blend with blurred white layer
        if random.random() < self.fog_prob:
            img = self._add_fog(img)
        # Sun glare: increase brightness in a region
        if random.random() < self.sun_prob:
            img = self._add_sun_glare(img)
        return img

    def _add_rain(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        h, w = arr.shape
        # create rain streaks
        rain = np.zeros((h, w), dtype=np.uint8)
        num_drops = int(h * w * 0.0005)
        for _ in range(num_drops):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            length = random.randint(10, 20)
            thickness = random.randint(1, 2)
            for i in range(length):
                yy = min(h-1, y+i)
                rain[yy, x] = 255
        rain_img = Image.fromarray(rain).filter(ImageFilter.GaussianBlur(radius=1))
        return Image.blend(img, rain_img.convert('L'), alpha=0.3)

    def _add_fog(self, img: Image.Image) -> Image.Image:
        fog_layer = Image.new('L', img.size, color=255)
        fog_layer = fog_layer.filter(ImageFilter.GaussianBlur(radius=img.size[0]//15))
        return Image.blend(img, fog_layer, alpha=0.4)

    def _add_sun_glare(self, img: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Brightness(img)
        # brighten a circular region
        w, h = img.size
        mask = Image.new('L', (w, h), 0)
        draw = Image.new('L', (w, h), 0)
        cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        radius = random.randint(min(w,h)//8, min(w,h)//4)
        yy, xx = np.ogrid[:h, :w]
        circle = ((xx - cx)**2 + (yy - cy)**2) <= radius**2
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        mask_arr[circle] = 255
        mask = Image.fromarray(mask_arr)
        bright = enhancer.enhance(1.5)
        return Image.composite(bright, img, mask)

# Dataset loader for Avenue
class AvenueDataset(Dataset):
    def __init__(self, root, phase='training', transform=None, gt_list=None):
        self.phase = phase
        self.root = root
        self.transform = transform
        self.paths, self.labels = [], []
        base = os.path.join(root, phase, 'frames')
        vids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        for vid in vids:
            frame_dir = os.path.join(base, vid)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp'):
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
        if self.phase == 'testing':
            return x, self.labels[idx]
        return x

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
    decoder_params = list(model[1].parameters())
    optimizer = optim.Adam(decoder_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = CombinedLoss()
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
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
    ds = AvenueDataset(root, phase='testing', transform=test_transform, gt_list=gt_list)
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



def main(dataset_path):
    root = dataset_path
    # define transforms
    train_transform = transforms.Compose([
        EnvironmentalTransform(rain_prob=0.3, fog_prob=0.3, sun_prob=0.3),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    global test_transform
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    gt_list = load_avenue_gt(root)
    train_ds = AvenueDataset(root, phase='training', transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    model = convnext_autoencoder()
    model = train_model(model, train_loader, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), 'convnext_avenue_env.pth')
    evaluate(model, root, gt_list)
    
if __name__ == '__main__':
    

    # Set up argument parser
    parser = argparse.ArgumentParser(description=".")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
