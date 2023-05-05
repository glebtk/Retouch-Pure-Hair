import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transforms:
    def __init__(self, image_size):
        self.add_noise = A.Compose([
            A.GaussNoise(var_limit=(0, 200), p=0.2),
            A.ImageCompression(quality_lower=90, quality_upper=100, compression_type=1, p=0.2),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.2)
        ])

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=1.0,
                               shift_limit=(-0.25, 0.25),
                               scale_limit=(-0.3, 0.3),
                               rotate_limit=(-360, 360),
                               interpolation=2,
                               border_mode=0),
            A.Resize(image_size, image_size),
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
