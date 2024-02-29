import argparse
import importlib
import os

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from mc_to_sky.data_utils import build_dataset
from mc_to_sky.utils.train_utils import build_model, get_exp_dir, backup_script
from mc_to_sky.utils.yaml_utils import dump_yaml, read_yaml

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"


def get_parser():
    parser = argparse.ArgumentParser(description="Example argparse program")
    parser.add_argument("--config", "-y", type=str, help="path to config file")
    parser.add_argument("--ckpt_path", "-c", type=str, default=None, help="path to ckpt file for restore training")
    parser.add_argument("--load_weight_only", action="store_true", help="only load weight from ckpt for model")
    args = parser.parse_args()
    return args


def get_callback():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch}-{val_loss:.2f}",  # checkpoint file format
        save_top_k=1,  # save best checkpoint
        save_last=True,  # save last checkpoint
        mode="min",  # 'min' val_loss
    )
    return checkpoint_callback


def main():
    args = get_parser()
    hypes = read_yaml(args.config)
    train_conf = hypes["train_conf"]

    train_set = build_dataset(hypes, split="train")
    valid_set = build_dataset(hypes, split="val")

    train_loader = DataLoader(
        train_set, batch_size=train_conf["batch_size"], shuffle=True, num_workers=24, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=train_conf["batch_size"], shuffle=False, num_workers=24, pin_memory=True
    )

    # set up training
    if args.ckpt_path and not args.load_weight_only:
        exp_dir = args.ckpt_path.split("lightning_logs")[0]
    else:
        exp_dir = get_exp_dir(hypes["exp_name"])
        dump_yaml(hypes, exp_dir)

    backup_script(exp_dir)

    model = build_model(hypes)
    if args.load_weight_only:
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
        args.ckpt_path = None

    checkpoint_callback = get_callback()
    trainer = pl.Trainer(
        default_root_dir=exp_dir,
        accelerator=train_conf["accelerator"],
        devices=train_conf["device_num"],
        max_epochs=train_conf["epoch"],
        check_val_every_n_epoch=train_conf[
            "check_val_every_n_epoch"
        ],  # val and save at the same time
        log_every_n_steps=train_conf[
            "log_every_n_steps"
        ],  # , # default tensorboard log
        callbacks=[checkpoint_callback],
    )

    # start training
    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
