import os
import argparse
import yaml
from PIL import Image
import torch
from torchvision.utils import save_image
from utils import load_checkpoint, make_directory
from models import Model
from transforms import Transforms
from dataset import read_image


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")


def inference(image_path: str, model_name="model_checkpoint.pth.tar", return_input=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    check_file_exists("config.yaml")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    check_file_exists(image_path)
    image = read_image(image_path)

    model_checkpoint_path = os.path.join(config.get("checkpoint_dir", ""), model_name)
    check_file_exists(model_checkpoint_path)
    checkpoint = load_checkpoint(model_checkpoint_path, device=device)

    model = checkpoint["model"]
    model.eval()

    transforms = Transforms(image_size=256).test_transforms

    image_tensor = transforms(image=image)["image"].unsqueeze(0).to(device) / 255
    output = model(image_tensor)

    input = (image_tensor * 255).clamp(0, 255).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    output = (output * 255).clamp(0, 255).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    output_image = Image.fromarray(output.astype("uint8"))

    output_path = os.path.join("output", f"{os.path.splitext(os.path.basename(image_path))[0]}_result.png")
    make_directory("output")
    output_image.save(output_path)

    if return_input:
        return input, output
    else:
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_name", type=str, default="model_checkpoint.pth.tar", help="Name of the model checkpoint.")
    parser.add_argument("--return_input", type=bool, default=False, help="Return input images for comparation")
    args = parser.parse_args()

    try:
        inference(args.image_path, args.model_name, args.return_input)
    except Exception as e:
        print(f"Error: {e}")
