import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(size=512):
    return A.Compose([
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(clip_limit=2),
        ], p=0.6),
        A.GaussianBlur(blur_limit=(3,7), p=0.2),
        A.Resize(size, size),
        A.Normalize(),  # uses ImageNet mean/std by default in ToTensorV2
        ToTensorV2(),
    ])

def get_valid_transforms(size=512):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(),
    ])
