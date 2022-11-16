import os
import sys
from argparse import Namespace
from typing import Any, Dict

import wandb
from scaler_gan.distributed.distributed import run_on_main
from scaler_gan.scalergan_utils.global_logger import logger
from scaler_gan.scalergan_utils.scalergan_utils import (
    G_PRED_DIR,
    G_PRED_PLT,
    MELS_IMGS_DIR,
    RECONSTRUCT,
    RECONSTURCT_PLT,
    save_g_pred_plot,
    save_reconstruct_plot,
)
from scaler_gan.trainer.scalerGAN import ScalerGANTrainer


class Visualizer:
    """
    Visualizer class
    """

    def __init__(self, gan: ScalerGANTrainer, conf: Namespace):
        """
        Init
        :param gan: Scaler GAN Trainer object
        :param conf: Configuration
        """
        # Init weights and biases logs:
        project_name = os.path.basename(conf.output_dir)
        run_on_main(
            self.init_wandb, kwargs={"conf": conf, "project_name": project_name}
        )
        self.gan = gan
        self.conf = conf

        self.g_pred_plt_dir = os.path.join(
            self.conf.artifacts_dir, MELS_IMGS_DIR, G_PRED_DIR
        )
        self.recunstruct_plt_dir = os.path.join(
            self.conf.artifacts_dir, MELS_IMGS_DIR, RECONSTRUCT
        )
        self.G_loss = [None] * conf.num_epochs
        self.D_loss_real = [None] * conf.num_epochs
        self.D_loss_fake = [None] * conf.num_epochs

        self.Rec_loss = [None] * conf.num_epochs
        self.logger = logger
        run_on_main(self.log_h_params)

    def init_wandb(self, conf: Namespace, project_name: str):
        """
        Init weights and biases
        Args:
            conf (Namespace): Configuration
            project_name (str): Project name
        """
        wandb.init(
            project="scalerGAN",
            entity="eyalcohen",
            name=project_name,
            config=vars(conf),
        )

    def close_wandb(self):
        """
        Finish wandb process
        :return:
        """
        run_on_main(wandb.finish)

    def log_h_params(self):
        """
        Log hyper parameters
        :return:
        """
        self.logger.info(
            "Arguments:\n"
            + "\n".join(f"{arg}: {getattr(self.conf, arg)}" for arg in vars(self.conf))
        )
        self.logger.info("Non default Arguments:")
        self.logger.info(
            "\n".join(
                f"{x.replace('-','')}: {y}"
                for x, y in zip(sys.argv[1::2], sys.argv[2::2])
            )
        )

    def log_results(self):
        """
        Log results and save plots.
        :return:
        """
        run_on_main(self._log_results)
        if not self.gan.cur_epoch % self.conf.save_snapshot_freq:
            # Save snapshot when needed
            self.gan.save_checkpoint()

        self.gan.cur_epoch += 1

    def _log_results(self):
        """
        log results with run_on_main function, to avoid multiple logs.
        """
        i = self.gan.cur_epoch
        input_mel = self.gan.input_tensor[0].cpu().data.numpy().squeeze()
        g_pred = self.gan.G_pred[0].cpu().data.numpy().squeeze()
        reconstruct = self.gan.reconstruct[0].cpu().data.numpy().squeeze()

        if not i % self.conf.reconsturct_plot_freq:
            plt_path = os.path.join(
                self.recunstruct_plt_dir, f"{RECONSTURCT_PLT}_{i}.jpg"
            )
            save_reconstruct_plot(input_mel, reconstruct, plt_path)

        if not i % self.conf.G_pred_freq_plot:
            plt_path = os.path.join(self.g_pred_plt_dir, f"{G_PRED_PLT}_{i}.jpg")
            save_g_pred_plot(input_mel, g_pred, plt_path)

        if not i % self.conf.print_epoch_freq:
            self.G_loss[i - self.conf.print_epoch_freq : i] = (
                self.gan.losses_G_gan.detach().cpu().float().numpy().tolist()
            )
            self.D_loss_real[i - self.conf.print_epoch_freq : i] = (
                self.gan.losses_D_real.detach().cpu().float().numpy().tolist()
            )
            self.D_loss_fake[i - self.conf.print_epoch_freq : i] = (
                self.gan.losses_D_fake.detach().cpu().float().numpy().tolist()
            )
            self.Rec_loss[i - self.conf.print_epoch_freq : i] = (
                self.gan.losses_G_reconstruct.detach().cpu().float().numpy().tolist()
            )

            results_dict = {
                "Epoch": i,
                "G_loss": self.G_loss[i - 1],
                "D_loss_real": self.D_loss_real[i - 1],
                "D_loss_fake": self.D_loss_fake[i - 1],
            }
            results_dict["Rec_loss"] = self.Rec_loss[i - 1]
            results_dict["LR"] = self.gan.lr_scheduler_G.get_last_lr()
            log_results_params(results_dict)


def log_results_params(results_dict: Dict[str, Any]):
    """
    Log results params
    :param results_dict: Results dictionary 
    :return:
    """
    msg = ", ".join(
        f"{title}: {val:.6f}" if isinstance(val, float) else f"{title}: {val}"
        for title, val in results_dict.items()
    )
    logger.info(msg)
    wandb.log(results_dict)
