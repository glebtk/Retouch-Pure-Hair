import os
import cv2
import torch
import random
import numpy as np
from transforms import Transforms
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


def save_checkpoint(checkpoint, filepath):
    if not filepath.endswith(".pth.tar"):
        raise ValueError("Filepath should end with .pth.tar extension")

    try:
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        raise


def load_checkpoint(filepath, device):
    if not filepath.endswith(".pth.tar"):
        raise ValueError("Filepath should end with .pth.tar extension")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    try:
        checkpoint = torch.load(filepath, map_location=device)
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def get_last_checkpoint(checkpoint_dir):
    try:
        checkpoints = os.listdir(checkpoint_dir)
        checkpoints = [c for c in checkpoints if c.endswith(".pth.tar")]
        checkpoints.sort()

        return os.path.join(checkpoint_dir, checkpoints[-1])
    except IndexError:
        print(f"Error: there are no saved checkpoints in the {checkpoint_dir} directory")
        raise


def make_directory(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print(f"Error: directory \"{folder_path}\" already exists")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_metrics(true_images, denoised_images):
    mse_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    num_images = len(true_images)

    for true_img, denoised_img in zip(true_images, denoised_images):
        mse = torch.mean((true_img - denoised_img) ** 2)
        mse_total += mse.item()

        max_pixel_value = torch.max(true_img)
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        psnr_total += psnr.item()

        ssim_total += ssim(np.transpose(true_img.cpu().numpy(), (1, 2, 0)),
                           np.transpose(denoised_img.cpu().numpy(), (1, 2, 0)),
                           channel_axis=2,
                           data_range=max_pixel_value.item())

    return {
        "MSE": mse_total / num_images,
        "PSNR": psnr_total / num_images,
        "SSIM": ssim_total / num_images
    }


def test_model(checkpoint, data_loader, device, get_sample=False):
    noisy_images = []
    clean_images = []
    denoised_images = []

    checkpoint["generator"].eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for noisy_imgs, clean_imgs in data_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # Denoising
            noise = checkpoint["generator"](noisy_imgs)
            denoised_imgs = noisy_imgs - noise

            # Append images to the lists
            noisy_images.extend(noisy_imgs.cpu())
            clean_images.extend(clean_imgs.cpu())
            denoised_images.extend(denoised_imgs.cpu())

    checkpoint["generator"].train()

    metrics = get_metrics(true_images=clean_images, denoised_images=denoised_images)

    if get_sample:
        num = random.randint(0, len(noisy_imgs) - 1)
        sample = {
            "noisy": postprocessing(noisy_imgs[num]),
            "clean": postprocessing(clean_imgs[num]),
            "denoised": postprocessing(denoised_imgs[num])
        }

        return metrics, sample
    else:
        return metrics


def postprocessing(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    if len(image.shape) == 4:
        assert len(image.shape) == 3, "Input must be a single image, not a batch of images"

    return (image * 255).astype('uint8')


def log_tensorboard(losses, metrics, writer, global_step, sample):
    # Log losses
    for loss_name, loss_value in losses.items():
        writer.add_scalar(f"Loss/{loss_name}", loss_value, global_step)

    # Log metrics
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f"Metrics/{metric_name}", metric_value, global_step)

    # Log sample images
    sample_images = np.concatenate((sample["noisy"], sample["clean"], sample["denoised"]), axis=2)
    writer.add_image("Sample_images", sample_images, global_step)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set as {seed}")
