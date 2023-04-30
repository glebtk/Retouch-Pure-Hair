import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def read_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class HairDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transform=None):
        self.root_dir = root_dir
        self.data_csv = pd.read_csv(os.path.join(root_dir, csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):
        def open_transform_image(input_path, target_path):
            input_img = read_image(input_path)
            target_img = read_image(target_path)

            if self.transform:
                transformed = self.transform(
                    image=input_img,
                    image0=target_img
                )
                return transformed["image"], transformed["image0"]
            else:
                to_tensor = ToTensorV2()
                return to_tensor(input_img), to_tensor(target_img)

        img_paths = [os.path.join(self.root_dir, self.data_csv.iloc[index, i]) for i in range(2)]
        return open_transform_image(*img_paths)


def dataset_test():
    root_dir = "./dataset/train"
    csv_file = "labels.csv"
    dataset = HairDataset(root_dir=root_dir, csv_file=csv_file)

    print("Done!")


if __name__ == "__main__":
    dataset_test()
