import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transforms:
    def __init__(self, image_size, mean, std):
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=1.0),
            A.RandomScale(scale_limit=(0.5, 1.5)),
            A.RandomCrop(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ],
            additional_targets={"image0": "image"}
        )

        self.test_transforms = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            A.Resize(image_size, image_size),
            ToTensorV2()
        ],
            additional_targets={"image0": "image"}
        )
