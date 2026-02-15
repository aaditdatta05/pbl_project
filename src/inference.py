import torch
from torch.utils.data import DataLoader
from dataset import DRDataset
from model import DRModelV2
from transforms import get_valid_transforms
import pandas as pd
from utils import load_state_dict_compatible

def run_inference(model_path, csv_file, images_dir, out_csv='preds.csv', device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = DRModelV2(pretrained=False)
    ckpt = torch.load(model_path, map_location='cpu')
    _, skipped = load_state_dict_compatible(model, ckpt['model_state'])
    if skipped:
        print(f"Skipped {len(skipped)} incompatible keys from checkpoint.")
    model.to(device)
    model.eval()

    ds = DRDataset(csv_file, images_dir, transforms=get_valid_transforms(512), mode='test')
    loader = DataLoader(ds, batch_size=8, num_workers=4)

    all_preds = []
    all_probs = []
    ids = []
    with torch.no_grad():
        for imgs, image_ids in loader:
            imgs = imgs.to(device)
            grades, probs = model.predict_ordinal(imgs)
            all_preds.extend(grades.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            ids.extend(image_ids)

    out_df = pd.DataFrame({
        'image_id': ids,
        'prediction': all_preds,
        'cum_probs': all_probs
    })
    out_df.to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__ == '__main__':
    # example usage
    run_inference('outputs/checkpoints/best.pth', 'data/test_clean.csv', 'data/test_images', out_csv='outputs/preds.csv')
