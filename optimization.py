import os
import sys
import yaml
import torch
import signal
import joblib
import optuna
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from train import get_config, train
from transforms import Transforms
from dataset import HairDataset
from models import Model
from tqdm import tqdm
from utils import *


def objective_model(trial):
    config = get_config()
    config.num_epochs = 10

    config.weight_init = trial.suggest_categorical('weight_init', [None, 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'])
    config.num_layers = trial.suggest_int('num_layers', 1, 10)
    config.num_features = trial.suggest_int('num_features', 20, 80)

    train_name = f"optim_model_weights={config.weight_init}_layers={config.num_layers}_features={config.num_features}"
    config.train_name = train_name

    metrics = main(config)
    return metrics["PSNR"]


def objective_train(trial):
    config = get_config()
    config.num_epochs = 10

    config.batch_size = trial.suggest_int('batch_size', 32, 128)
    config.learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    train_name = f"optim_hp_bs={config.batch_size}_lr={config.learning_rate}"
    config.train_name = train_name

    metrics = main(config)
    return metrics["PSNR"]


def main(config):
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

    metrics = train(checkpoint, train_data_loader, test_data_loader, device, config)
    return metrics


def load_and_combine_studies(study_files):
    studies = []
    for file in study_files:
        studies.append(joblib.load(file))

    combined_study = optuna.study.create_study(direction="maximize")
    for study in studies:
        for trial in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
            combined_study.add_trial(trial)

    return combined_study


def save_and_exit(signal_number, frame):
    joblib.dump(study, f"study/study_{phase}_combined.pkl")
    print(f"Study сохранен в study/study_{phase}_combined.pkl.")
    sys.exit(0)


if __name__ == "__main__":
    phase = "optim_hp"

    make_directory("./study")

    study_files = [os.path.join("study", name) for name in os.listdir("study")]

    if all(os.path.exists(file) for file in study_files):
        study = load_and_combine_studies(study_files)
        print(f"Загружены и объединены сохраненные Study для фазы {phase}.")
    else:
        study = optuna.create_study(direction='maximize')
        print(f"Создано новое Study для фазы {phase}.")

    signal.signal(signal.SIGINT, save_and_exit)

    if phase == "optim_model":
        study.optimize(objective_model, n_trials=100)

        best_trial = study.best_trial
        print(f"Best trial: {best_trial.value}, params: {best_trial.params}")
    elif phase == "optim_hp":
        study.optimize(objective_train, n_trials=100)

        best_trial = study.best_trial
        print(f"Best trial: {best_trial.value}, params: {best_trial.params}")

    joblib.dump(study, f"study/study_{phase}_combined.pkl")
    print(f"Study сохранен в study/study_{phase}_combined.pkl.")

