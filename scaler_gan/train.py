from argparse import Namespace
import os
import sys
import git


wt_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
try:
    sys.path.index(wt_dir)
except ValueError:
    sys.path.append(wt_dir)

from scaler_gan.scalergan_utils.scalergan_utils import init_logger
from torch.utils.data import DataLoader, Subset
from scaler_gan.configs.configs import Config
from scaler_gan.data_generator.dataloader import MelDataset
from scaler_gan.distributed import distributed
from scaler_gan.scalergan_utils.visualizer.visual import Visualizer
from scaler_gan.trainer.scalerGAN import ScalerGANTrainer


def train(conf: Namespace):
    # Initialize the logger
    log_file_path = os.path.join(conf.artifacts_dir, "train.log")
    log_level = "DEBUG" if conf.verbose or conf.debug else "INFO"
    distributed.run_on_main(
        init_logger, kwargs={"log_file": log_file_path, "log_level": log_level}
    )

    # Initialize the trainer
    gan = ScalerGANTrainer(conf)

    # Initialize the visualizer
    visualizer = Visualizer(gan, conf)

    # Data preparation
    train_set = MelDataset(conf.input_file, conf.must_divide, **conf.mel_params)
    if conf.debug:
        if conf.distributed:
            visualizer.logger.info(
                "Debug mode can't be used with distributed training, skiping debug mode"
            )
        else:
            visualizer.logger.info("Debug mode is on. Will run only on 50 examples")
            train_set = Subset(train_set, list(range(50)))

    if conf.distributed:
        # Initialize distributed training
        distributed.ddp_init_group(
            conf.distributed, conf.distributed_backend, conf.local_rank
        )
        train_loader_kwargs = {
            "num_workers": conf.num_workers,
            "batch_size": conf.batch_size,
            "pin_memory": True,
            "drop_last": True,
        }
        train_loader = distributed.loader(
            train_set, shuffle=True, dataloader_class=DataLoader, **train_loader_kwargs
        )
        gan.wrap_ddp()
    else:
        # Initialize single process training
        train_loader = DataLoader(
            train_set,
            num_workers=conf.num_workers,
            shuffle=True,
            batch_size=conf.batch_size,
            pin_memory=True,
            drop_last=True,
        )

    # ====== Training ====== #
    for _ in range(gan.cur_epoch, conf.num_epochs + 1):
        # Train the model for one epoch
        gan.train_one_epoch(train_loader)
        # Log the model results.
        visualizer.log_results()

    visualizer.close_wandb()


def main():
    """
    The main process for training the Scaler GAN
    :return:
    """
    # Loads argparser config
    conf = Config().parse()
    train(conf)


if __name__ == "__main__":
    main()
