import yaml
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from models import Generator, Discriminator
from transforms import Transforms
from dataset import HairDataset
from tqdm import tqdm
from utils import *


def train_one_epoch(checkpoint, data_loader, device, writer, config):
    checkpoint["generator"].train()
    checkpoint["discriminator"].train()

    MSE = nn.MSELoss()

    g_scaler = GradScaler()
    d_scaler = GradScaler()

    # Training loop:
    loop = tqdm(data_loader)
    for idx, (input_image, target_image) in enumerate(loop):
        global_step = (checkpoint["epoch"] * len(data_loader) + idx) * len(input_image)

        input_image = input_image.to(device)
        target_image = target_image.to(device)

        # ---------- Train the generator ---------- #
        with autocast():  # Enable mixed precision
            # Generate reconstructed images using the generator
            reconstructed_image = checkpoint["generator"](input_image)

            # Calculate MSE and adversarial losses for the generator
            generator_mse_loss = MSE(target_image, reconstructed_image)
            disc_pred = checkpoint["discriminator"](reconstructed_image.detach()).mean()
            generator_adv_loss = MSE(disc_pred, torch.ones_like(disc_pred))

            # Compute the total generator loss
            generator_loss = generator_mse_loss + generator_adv_loss * config.adv_weight

        # Perform backpropagation and update the generator weights
        checkpoint["opt_gen"].zero_grad()
        g_scaler.scale(generator_loss).backward()
        g_scaler.step(checkpoint["opt_gen"])
        g_scaler.update()

        # ---------- Train the discriminator ---------- #
        with autocast():
            # Generate output images using the generator
            reconstructed_image = checkpoint["generator"](input_image.detach())

            # Get discriminator predictions for real and generated images
            target_disc_pred = checkpoint["discriminator"](target_image.detach())
            recon_disc_pred = checkpoint["discriminator"](reconstructed_image.detach())

            # Calculate the losses for real and generated images
            target_disc_loss = MSE(target_disc_pred, torch.ones_like(target_disc_pred))
            recon_disc_loss = MSE(recon_disc_pred, torch.zeros_like(recon_disc_pred))

            # Compute the total discriminator loss
            discriminator_loss = (target_disc_loss + recon_disc_loss) / 2

        # Perform backpropagation and update the discriminator weights
        checkpoint["opt_disc"].zero_grad()
        d_scaler.scale(discriminator_loss).backward()
        d_scaler.step(checkpoint["opt_disc"])
        d_scaler.update()

        # Updating tensorboard (current fake images)
        if idx % 32 == 0:
            input_image = postprocessing(input_image, config)
            target_image = postprocessing(target_image, config)
            fake_image = postprocessing(reconstructed_image, config)
            current_images = np.concatenate((input_image, target_image, fake_image), axis=2)
            writer.add_image(f"Current images", current_images, global_step=global_step)

            writer.add_scalar("Generator MSE loss", generator_mse_loss.item(), global_step=global_step)
            writer.add_scalar("Generator adversarial loss", generator_adv_loss.item(), global_step=global_step)
            writer.add_scalar("Discriminator loss", discriminator_loss.item(), global_step=global_step)

    checkpoint["epoch"] += 1


def train(checkpoint, data_loader, device, config):
    train_name = get_current_time()
    writer = SummaryWriter(f"tb/train_{train_name}")

    for epoch in range(checkpoint["epoch"], config.num_epochs):

        train_one_epoch(checkpoint, data_loader, device, writer, config)

        # Save checkpoint
        if config.save_checkpoint:
            print("=> Saving a checkpoint")
            save_checkpoint(checkpoint, os.path.join(config.checkpoint_dir, f"checkpoint_{train_name}_{epoch}.pth.tar"))

        # Updating tensorboard (test images)
        # writer.add_image("Test images", model_test(checkpoint["generator"], config, device), global_step=epoch)


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
        set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading the dataset
    transforms = Transforms(config.image_size, config.dataset_mean, config.dataset_std)
    dataset = HairDataset(
        root_dir=config.dataset,
        csv_file="labels.csv",
        transform=transforms.train_transforms
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Loading the latest checkpoint models
    if config.load_checkpoint:
        print("=> Loading the last checkpoint")

        checkpoint_path = get_last_checkpoint(config.checkpoint_dir)
        checkpoint = load_checkpoint(checkpoint_path, device)
    else:
        generator = Generator(in_channels=config.in_channels,
                              out_channels=config.out_channels,
                              embedding_size=config.embedding_size,
                              weight_init=config.weight_init).to(device)

        discriminator = Discriminator(in_channels=config.out_channels,
                                      weight_init=config.weight_init).to(device)

        opt_gen = optim.Adam(
            params=list(generator.parameters()),
            lr=config.generator_learning_rate
        )
        opt_disc = optim.Adam(
            params=list(discriminator.parameters()),
            lr=config.discriminator_learning_rate
        )

        checkpoint = {
            "generator": generator,
            "discriminator": discriminator,
            "opt_gen": opt_gen,
            "opt_disc": opt_disc,
            "epoch": 0
        }

    train(checkpoint, data_loader, device, config)


if __name__ == "__main__":
    main()
