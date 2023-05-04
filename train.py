import os
import yaml
import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from models import Model
from transforms import Transforms
from dataset import HairDataset
from tqdm import tqdm
from utils import *


def train_one_epoch(checkpoint, data_loader, device):
    MSE = nn.MSELoss()

    losses = {
        "generator_mse_loss": 0.0
    }

    scaler = GradScaler()

    checkpoint["model"].train()

    # Training loop:
    loop = tqdm(data_loader)
    for idx, (noisy_image, clean_image) in enumerate(loop):
        noisy_image = noisy_image.to(device)
        clean_image = clean_image.to(device)

        with autocast():  # Enable mixed precision
            denoised_image = checkpoint["model"](noisy_image)

            generator_mse_loss = MSE(clean_image, denoised_image)
            losses["mse"] += generator_mse_loss.item()

        # Perform backpropagation and update the generator weights
        checkpoint["opt"].zero_grad()
        scaler.scale(losses["mse"]).backward()
        scaler.step(checkpoint["opt"])
        scaler.update()

    losses = {k: v / len(data_loader) for k, v in losses.items()}
    return losses


def train(checkpoint, train_data_loader, test_data_loader, device, config):
    train_name = config.train_name

    if not train_name:
        train_name = get_current_time()

    with SummaryWriter(f"tb/train_{train_name}") as writer:
        for epoch in range(checkpoint["epoch"], config.num_epochs):

            losses = train_one_epoch(checkpoint, train_data_loader, device)
            metrics, sample = test_model(checkpoint, test_data_loader, device, get_sample=True)

            checkpoint["epoch"] += 1

            # Save checkpoint
            if config.save_checkpoint:
                print("=> Saving a checkpoint")
                save_checkpoint(checkpoint, config.checkpoint_dir, f"checkpoint_{train_name}_{epoch}.pth.tar")

            # Updating tensorboard
            log_tensorboard(losses, metrics, writer, global_step=epoch, sample=sample)


def get_config():
    # Load default configuration from YAML file
    with open("config.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    # Create argparse parser and add arguments
    parser = argparse.ArgumentParser(description="Train a model")

    for key, value in default_config.items():
        if isinstance(value, (int, float, str, bool)):
            parser.add_argument(f"--{key}", type=type(value), default=value)

    parser.set_defaults(**default_config)

    return parser.parse_args()


def main():
    config = get_config()

    if config.set_seed:
        set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading the dataset
    data = pd.read_csv(os.path.join(config.dataset, "labels.csv"))
    transforms = Transforms(config.image_size)

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=config.test_size, random_state=config.seed)

    train_dataset = HairDataset(
        root_dir=config.dataset,
        data=train_data,
        transform=transforms.train_transforms
    )
    test_dataset = HairDataset(
        root_dir=config.dataset,
        data=test_data,
        transform=transforms.test_transforms
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Loading the latest checkpoint models
    if config.load_checkpoint:
        print("=> Loading the last checkpoint")

        checkpoint_path = get_last_checkpoint(config.checkpoint_dir)
        checkpoint = load_checkpoint(checkpoint_path, device)
    else:
        model = Model(in_channels=config.in_channels,
                      out_channels=config.out_channels,
                      num_layers=config.num_layers,
                      num_features=config.num_features,
                      weight_init=config.weight_init).to(device)

        opt = optim.Adam(
            params=list(model.parameters()),
            lr=config.learning_rate,
            betas=(0.5, 0.999)
        )

        checkpoint = {
            "model": model,
            "opt": opt,
            "epoch": 0
        }

    train(checkpoint, train_data_loader, test_data_loader, device, config)


if __name__ == "__main__":
    main()
