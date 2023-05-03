import os
import zipfile
import urllib.request
import argparse


def download_and_unzip(url, path, name):
    full_path = os.path.join(path, name)

    try:
        urllib.request.urlretrieve(url, full_path)
    except FileNotFoundError:
        os.mkdir(path)
        urllib.request.urlretrieve(url, full_path)

    dataset_zip = zipfile.ZipFile(full_path, 'r')
    dataset_zip.extractall(path)


def download_files(load_dataset, load_checkpoint):
    if load_dataset:
        url = "https://gitlab.com/glebtutik/pure_hair_data/-/raw/main/hair_dataset.zip"
        path = "./dataset"
        file_name = "dataset.zip"

        download_and_unzip(url, path, file_name)
        os.remove(os.path.join(path, file_name))

        print("=> The dataset is loaded!")

    if load_checkpoint:
        # url = "https://gitlab.com/glebtutik/crimean_plants_classification_files/-/raw/main/checkpoints/model_checkpoint.zip"
        # path = "checkpoints"
        # file_name = "model_checkpoint.zip"
        #
        # download_and_unzip(url, path, file_name)
        # os.remove(os.path.join(path, file_name))

        # print("=> The checkpoint is loaded!")
        print("=> Checkpoint is not available now :(")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and unzip dataset and/or checkpoint")
    parser.add_argument("--load-dataset", action="store_true", help="Download and unzip dataset")
    parser.add_argument("--load-checkpoint", action="store_true", help="Download and unzip checkpoint")
    args = parser.parse_args()

    download_files(args.load_dataset, args.load_checkpoint)
