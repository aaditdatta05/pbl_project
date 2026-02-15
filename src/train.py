import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from dataset import DRDataset
from model import DRModelV2
from transforms import get_train_transforms, get_valid_transforms
from losses import OrdinalRegressionLoss
from utils import round_predictions, quadratic_weighted_kappa, compute_pos_weight_from_labels
import pandas as pd

def train_one_epoch(model, loader, optimizer, device, scaler, criterion):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader):
        imgs = imgs.to(device)
        targets = targets.to(device).long()  # Ordinal labels: 0-4
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(imgs)  # Shape: (batch_size, 4) cumulative logits
            loss = criterion(outputs, targets)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for imgs, targets in tqdm(loader):
            imgs = imgs.to(device)
            # Use predict_ordinal to get class predictions directly
            grades, probs = model.predict_ordinal(imgs)
            preds.extend(grades.detach().cpu().numpy())
            gts.extend(targets.numpy())
    preds = np.array(preds)
    gts = np.array(gts)
    qwk = quadratic_weighted_kappa(gts, preds)
    return qwk, preds, gts

def main():
    # Config (change paths as needed)
    train_csv = 'data/train_split.csv'
    images_dir = 'data/train_images'
    val_csv = 'data/val.csv'
    model_save = 'outputs/checkpoints/best.pth'
    os.makedirs(os.path.dirname(model_save), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # EfficientNetV2-S with ordinal regression
    model = DRModelV2(pretrained=True, num_classes=5)
    model.to(device)

    train_df = pd.read_csv(train_csv)
    if 'label' not in train_df.columns:
        raise ValueError("train_csv must include a 'label' column for class weights.")
    pos_weight = compute_pos_weight_from_labels(
        train_df['label'].values,
        num_classes=5,
        beta=0.9999
    )
    print(f"Ordinal pos_weight per threshold: {pos_weight.tolist()}")

    train_ds = DRDataset(train_csv, images_dir, transforms=get_train_transforms(512), mode='train')
    val_ds = DRDataset(val_csv, images_dir, transforms=get_valid_transforms(512), mode='train')

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    criterion = OrdinalRegressionLoss(num_classes=5, pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler()

    best_qwk = -1
    for epoch in range(1, 31):
        print(f"Epoch {epoch}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)
        qwk, preds, gts = validate(model, val_loader, device)
        print(f"Train loss {train_loss:.4f} — Val QWK {qwk:.4f}")
        scheduler.step()
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, model_save)
            print("Saved best model.")
    print("Training finished.")

if __name__ == '__main__':
    main()
