import albumentations as A
from albumentations.pytorch import ToTensorV2


import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transforms:
    def __init__(self, image_size):
        self.add_noise = A.Compose([
            A.GaussNoise(var_limit=(0, 321), p=1),
            A.JpegCompression(quality_lower=60, quality_upper=100, p=1),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1)
        ])

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-180, 180), p=1.0),
            A.RandomScale(scale_limit=(-0.6, 0.6)),
            A.RandomCrop(height=image_size, width=image_size),
            ToTensorV2()
        ],
            additional_targets={"image0": "image"}
        )

        self.test_transforms = A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ],
            additional_targets={"image0": "image"}
        )
