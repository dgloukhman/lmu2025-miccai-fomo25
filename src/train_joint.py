#!/usr/bin/env python

import os
import torch
import lightning as L
import argparse
import warnings
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)

from models.joint_mae_seg import JointMaeSegModel
from augmentations.augmentation_composer import (
    get_pretrain_augmentations,
    get_val_augmentations,
)
from data.datamodule import JointMaeSegDataModule
from yucca.pipeline.configuration.split_data import get_split_config
from yucca.pipeline.configuration.configure_paths import detect_version
from utils.utils import setup_seed, SimplePathConfig
from data.task_configs import task2_config # Using Task2 for segmentation


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to output directory for models and logs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument("--model_name", type=str, default="unet_b")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--new_version", action="store_true")
    parser.add_argument("--split_method", type=str, default="simple_train_val_split")
    parser.add_argument("--split_param", type=str, help="Split parameter", default=0.2)
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument("--experiment", type=str, default="joint_experiment")

    args = parser.parse_args()

    task_cfg = task2_config
    task_name = task_cfg["task_name"]
    train_data_dir = os.path.join(args.data_dir, task_name)

    save_dir = os.path.join(args.save_dir, "models", task_name, args.model_name)
    versions_dir = os.path.join(save_dir, "versions")
    continue_from_most_recent = not args.new_version
    version = detect_version(versions_dir, continue_from_most_recent)
    version_dir = os.path.join(versions_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    seed = setup_seed(continue_from_most_recent)

    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    splits_config = get_split_config(
        method=args.split_method,
        param=float(args.split_param),
        path_config=path_config,
    )

    config = {
        "experiment": args.experiment,
        "model_name": args.model_name,
        "version": version,
        "save_dir": save_dir,
        "train_data_dir": train_data_dir,
        "version_dir": version_dir,
        "seed": seed,
        "patch_size": (args.patch_size,) * 3,
        "num_classes": task_cfg["num_classes"],
        "num_modalities": len(task_cfg["modalities"]),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "precision": args.precision,
        "num_devices": args.num_devices,
        "num_workers": args.num_workers,
        "fast_dev_run": args.fast_dev_run,
    }

    train_transforms = get_pretrain_augmentations(config["patch_size"], "none")
    val_transforms = get_val_augmentations()

    data = JointMaeSegDataModule(
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        splits_config=splits_config,
        split_idx=args.split_idx,
        train_data_dir=train_data_dir,
        composed_train_transforms=train_transforms,
        composed_val_transforms=val_transforms,
    )

    model = JointMaeSegModel(config=config, learning_rate=args.learning_rate)

    wandb.init(
        project="fomo-joint-training",
        name=f"{config['experiment']}_version_{config['version']}",
    )

    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="fomo-joint-training",
        name=f"{config['experiment']}_version_{config['version']}",
        log_model=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=version_dir,
        filename="{epoch:02d}",
        every_n_epochs=25,
        save_last=True,
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if config["num_devices"] > 1 else "auto",
        num_nodes=1,
        devices=config["num_devices"],
        default_root_dir=config["save_dir"],
        max_epochs=config["epochs"],
        precision=config["precision"],
        fast_dev_run=config["fast_dev_run"],
    )

    trainer.fit(model=model, datamodule=data, ckpt_path="last")
    wandb.finish()

if __name__ == "__main__":
    main()
