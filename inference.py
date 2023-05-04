import os
import uuid
import argparse
import yaml
from PIL import Image
import torch
from torchvision.utils import save_image
from utils import load_checkpoint, make_directory
from models import Model
from transforms import Transforms
from dataset import read_image


def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_checkpoint_path = os.path.join(config["checkpoint_dir"], args.model_name)
    checkpoint = load_checkpoint(model_checkpoint_path, device=device)

    model = checkpoint["model"]
    model.eval()

    transforms = Transforms(image_size=256).test_transforms
    image = read_image(args.image_path)

    image_tensor = transforms(image=image)["image"].unsqueeze(0).to(device) / 255
    output = model(image_tensor)

    output = (output * 255).clamp(0, 255).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="image.png")
    parser.add_argument("--model_name", type=str, default="model_checkpoint.pth.tar")
    args = parser.parse_args()

    inference(args)
