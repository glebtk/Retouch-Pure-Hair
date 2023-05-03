import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def read_image(path: str) -> cv2.Mat:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class HairDataset(Dataset):
    def __init__(self, root_dir, data, transform=None):
        self.root_dir = root_dir
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_path, target_path = self._get_image_paths(index)
        input_img, target_img = self._open_transform_images(input_path, target_path)
        return input_img / 255.0, target_img / 255.0

    def _get_image_paths(self, index):
        return [os.path.join(self.root_dir, self.data.iloc[index, i]) for i in range(2)]

    def _open_transform_images(self, input_path, target_path):
        input_img = read_image(input_path)
        target_img = read_image(target_path)

        if self.transform:
            transformed = self.transform(image=input_img, image0=target_img)
            input_img = transformed["image"]
            target_img = transformed["image0"]
        else:
            to_tensor = ToTensorV2()
            input_img = to_tensor(image=input_img)["image"]
            target_img = to_tensor(image=target_img)["image"]

        return input_img, target_img


def dataset_test():
    root_dir = "./dataset"
    csv_file = "labels.csv"
    data = pd.read_csv(os.path.join(root_dir, csv_file))
    dataset = HairDataset(root_dir=root_dir, data=data)

    print("Done!")


if __name__ == "__main__":
    dataset_test()
