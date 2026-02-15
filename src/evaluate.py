import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    classification_report
)

from torch.utils.data import DataLoader
from src.dataset import DRDataset
from src.transforms import get_valid_transforms
from src.model import DRModelV2
from src.utils import load_state_dict_compatible

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def within_one_accuracy(y_true, y_pred):
    y_pred_round = np.clip(np.round(y_pred), 0, 4)
    diff = np.abs(y_true - y_pred_round)
    return np.mean(diff <= 1)

def save_confusion_matrix(y_true, y_pred, path="outputs/confusion_matrix_1.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks(range(5), range(5))
    plt.yticks(range(5), range(5))

    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def load_model(checkpoint_path, device):
    model = DRModelV2().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    _, skipped = load_state_dict_compatible(model, ckpt["model_state"])
    if skipped:
        print(f"Skipped {len(skipped)} incompatible keys from checkpoint.")
    model.eval()
    return model

def evaluate(
    val_csv="data/val.csv",
    img_dir="data/train_images",
    checkpoint="outputs/checkpoints/dr.pth",
    batch_size=16
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = load_model(checkpoint, device)

    print("Loading validation dataset...")
    tf = get_valid_transforms(512)
    val_ds = DRDataset(val_csv, img_dir, transforms=tf)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds = []
    all_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            grades, _ = model.predict_ordinal(imgs)
            all_preds.append(grades.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    pred_round = np.clip(all_preds, 0, 4).astype(int)

    results = {
        "QWK": float(qwk(all_targets, pred_round)),
        "MAE": float(mean_absolute_error(all_targets, pred_round)),
        "RMSE": float(np.sqrt(mean_squared_error(all_targets, pred_round))),
        "Within_1_Accuracy": float(within_one_accuracy(all_targets, pred_round)),
        "Accuracy": float(np.mean(pred_round == all_targets)),
        "Per-Class Report": classification_report(
            all_targets, pred_round, output_dict=True
        )
    }

    with open("outputs/eval_metrics_1.json", "w") as f:
        json.dump(results, f, indent=4)

    save_confusion_matrix(all_targets, pred_round, path="outputs/confusion_matrix_1.png")

    print("\n===== Evaluation Complete =====\n")
    print(json.dumps(results, indent=4))
    print("\nConfusion matrix saved → outputs/confusion_matrix_1.png\n")
    print("Full metrics saved → outputs/eval_metrics_1.json\n")

    return results

if __name__ == "__main__":
    evaluate()