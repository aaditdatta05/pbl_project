import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class DRDataset(Dataset):
    """
    Handles train or test mode. If csv has no 'label' column, returns image only.
    Expects CSV with columns: 'image_id' and optional 'label'
    """
    def __init__(self, csv_file, images_dir, transforms=None, mode='train', img_ext='.png'):
        self.df = pd.read_csv(csv_file, dtype={'image_id': str})
        self.images_dir = images_dir
        self.transforms = transforms
        self.mode = mode
        self.img_ext = img_ext

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        path = os.path.join(self.images_dir, image_id + self.img_ext)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']
        # To tensor and channel-first will be done in collate or by transforms

        if 'label' in self.df.columns and self.mode != 'test':
            label = row['label']
            return img, np.float32(label)
        else:
            return img, image_id
