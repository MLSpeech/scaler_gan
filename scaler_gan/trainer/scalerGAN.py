import gc
import os
import re
import warnings
from argparse import Namespace
from typing import List, Optional, Tuple

import torch
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from scaler_gan.distributed.distributed import run_on_main

from scaler_gan.network_topology.networks import (
    GANLoss,
    Generator,
    MultiScaleDiscriminator,
    RandomCrop,
    WeightedMSELoss,
    weights_init,
)
from scaler_gan.scalergan_utils.scalergan_utils import get_scale_weights, random_size
from scaler_gan.scalergan_utils.global_logger import logger

CHECKPOINT_DIR = "checkpoints"


class LRPolicy(object):
    """
    Learning rate decay policy
    """

    def __init__(self, start: int, end: int, decay: Optional[bool] = True):
        """
        Init
        :param start: Start epoch
        :param end: End epoch
        :param decay: Flag to decay or not
        """
        self.start = start
        self.end = end
        self.decay = decay

    def __call__(self, cur_epoch: int) -> float:
        """
        Calculate the decay factor
        :param cur_epoch: Current epoch
        :return: Decay factor
        """
        return (
            1.0 - max(0.0, float(cur_epoch - self.start) / float(self.end - self.start))
            if self.decay
            else 1.0
        )


# noinspection PyAttributeOutsideInit
class ScalerGANTrainer:
    """
    Trainer class for Scaler GAN
    """

    def __init__(self, conf: Namespace, inference=False):
        """
        Init trainer
        Args:
            conf (Namespace): Configuration
            inference (bool, optional): Flag to indicate if this is for inference. Defaults to False.
        """
        # Acquire configuration
        self.conf = conf
        # Set device to GPUs if available
        self.device = (
            conf.device + f":{conf.local_rank}" if conf.local_rank and conf.distributed >= 0 else conf.device
        )
        self.find_unused_parameters = conf.find_unused_parameters
        if self.device == "cuda" and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))
        else:
            self.device = torch.device("cpu")
        
        self._init_models(conf)
        
        if self.conf.resume or inference or self.conf.fine_tune:
            checkpoint_path = self.conf.resume if self.conf.resume else self.conf.checkpoint_path
            assert checkpoint_path, "Checkpoint path is not provided"
            self.resume(checkpoint_path, inference=inference)

    def _init_models(self, conf):
        self._cur_epoch = 1
        self.num_epochs = conf.num_epochs

        self.input_tensor = torch.FloatTensor(
            1, 1, conf.input_crop_size, conf.input_crop_size
        ).to(self.device)
        self.real_example = torch.FloatTensor(
            1, 1, conf.output_crop_size, conf.output_crop_size
        ).to(self.device)

        # Define networks
        self.G = Generator(
            conf.G_base_channels,
            conf.G_num_resblocks,
            conf.G_num_downscales,
            conf.G_use_bias,
            conf.G_skip,
        )
        self.D = MultiScaleDiscriminator(
            conf.output_crop_size,
            self.conf.D_max_num_scales,
            self.conf.D_scale_factor,
            self.conf.D_base_channels,
        )
        self.criterionGAN = GANLoss()
        self.Reconstruct_loss = WeightedMSELoss(use_l1=conf.use_L1)
        self.RandCrop = RandomCrop(
            [conf.input_crop_size, conf.input_crop_size], must_divide=conf.must_divide
        )

        # Make all networks run on GPU
        self.G.to(self.device)
        self.D.to(self.device)
        self.criterionGAN.to(self.device)
        self.Reconstruct_loss.to(self.device)
        self.RandCrop.to(self.device)

        # Define loss function
        self.criterionReconstruction = self.Reconstruct_loss.forward

        # Keeping track of losses- prepare tensors
        self.losses_G_gan = torch.FloatTensor(conf.print_epoch_freq).to(self.device)
        self.losses_D_real = torch.FloatTensor(conf.print_epoch_freq).to(self.device)
        self.losses_D_fake = torch.FloatTensor(conf.print_epoch_freq).to(self.device)
        self.losses_G_reconstruct = torch.FloatTensor(conf.print_epoch_freq).to(
            self.device
        )
        self.losses_D_reconstruct = torch.FloatTensor(conf.print_epoch_freq).to(
            self.device
        )

        # Initialize networks
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        # Initialize optimizers
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999)
        )

        # Learning rate scheduler
        # First define linearly decaying functions (decay starts at a special epoch)
        if conf.lr_start_decay_epoch is None:
            lr_function = LRPolicy(0, 0, decay=False)
        else:
            start_decay = conf.lr_start_decay_epoch
            end_decay = conf.num_epochs
            lr_function = LRPolicy(start_decay, end_decay)
        # Define learning rate schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_function
        )
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D, lr_function
        )

    def wrap_ddp(self):
        """
        Wrapping the model for distributed data parallel launch
        :return:
        """
        if self.conf.distributed:
            if any(p.requires_grad for p in self.G.parameters()):
                self.G = SyncBatchNorm.convert_sync_batchnorm(self.G)
                self.G = DistributedDataParallel(
                    self.G,
                    device_ids=[self.device],
                    find_unused_parameters=self.find_unused_parameters,
                )
            if any(p.requires_grad for p in self.D.parameters()):
                self.D = SyncBatchNorm.convert_sync_batchnorm(self.D)
                self.D = DistributedDataParallel(
                    self.D,
                    device_ids=[self.device],
                    find_unused_parameters=self.find_unused_parameters,
                )

    def resume(self, resume_path: str, inference: Optional[bool] = False):
        """
        Resume train from checkpoint
        :param resume_path: Path to the checkpoint file
        :param inference: Inference mode flag
        :return:
        """
        run_on_main(logger.info, kwargs={"msg": f"Loading checkpoint from {resume_path}"})
        resume = torch.load(resume_path, map_location=self.device)
        missing = []
        if "G" in resume:
            self.G.load_state_dict(resume["G"])
        else:
            missing.append("G")
        if "D" in resume:
            self.D.load_state_dict(resume["D"])
        else:
            missing.append("D")
        if not inference:
            if "optim_G" in resume:
                self.optimizer_G.load_state_dict(resume["optim_G"])
            else:
                missing.append("optimizer G")
            if "optim_D" in resume:
                self.optimizer_D.load_state_dict(resume["optim_D"])
            else:
                missing.append("optimizer D")
            if "sched_G" in resume:
                self.lr_scheduler_G.load_state_dict(resume["sched_G"])
            else:
                missing.append("lr scheduler G")
            if "sched_D" in resume:
                self.lr_scheduler_D.load_state_dict(resume["sched_D"])
            else:
                missing.append("lr scheduler G")
            if "loss" in resume:
                self.criterionGAN.load_state_dict(resume["loss"])
            else:
                missing.append("GAN loss")

            epoch_name = "epoch" if "epoch" in resume else "iter"
            if epoch_name in resume:
                self.cur_epoch = (
                    resume[epoch_name]
                    if isinstance(resume[epoch_name], int)
                    else int(re.findall("\d+", resume[epoch_name])[-1])
                )
            else:
                missing.append(epoch_name)
        if len(missing):
            warnings.warn(
                "Missing the following state dicts from checkpoint: {}".format(
                    ", ".join(missing)
                )
            )
        logger.info(f"Resuming from epoch {self.cur_epoch}")

    def inference(
        self,
        input_tensor: torch.Tensor,
        output_size: List[int],
        rand_affine: List[float],
        input_size: List[int],
        run_d_pred: Optional[bool] = True,
        run_reconstruct: Optional[bool] = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run inference on input
        :param input_tensor: Input tensor
        :param output_size: Output size
        :param rand_affine: random factors
        :param input_size: input size
        :param run_d_pred: Flag to run discriminator or not, default is True
        :param run_reconstruct: Flag to run reconstructor or not, default is True
        :return: Tuple with the predictions
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            self.G_pred = self.G(
                input_tensor, output_size=output_size, random_affine=rand_affine,
            )
            if run_d_pred:
                scale_weights_for_output = get_scale_weights(
                    i=self.cur_epoch,
                    max_i=self.conf.D_scale_weights_epoch_for_even_scales,
                    start_factor=self.conf.D_scale_weights_sigma,
                    input_shape=self.G_pred.shape[2:],
                    min_size=self.conf.D_min_input_size,
                    num_scales_limit=self.conf.D_max_num_scales,
                    scale_factor=self.conf.D_scale_factor,
                )
                scale_weights_for_input = get_scale_weights(
                    i=self.cur_epoch,
                    max_i=self.conf.D_scale_weights_epoch_for_even_scales,
                    start_factor=self.conf.D_scale_weights_sigma,
                    input_shape=input_tensor.shape[2:],
                    min_size=self.conf.D_min_input_size,
                    num_scales_limit=self.conf.D_max_num_scales,
                    scale_factor=self.conf.D_scale_factor,
                )
                self.D_preds = [
                    self.D(input_tensor, scale_weights_for_input),
                    self.D(self.G_pred, scale_weights_for_output),
                ]
            else:
                self.D_preds = None

            self.reconstruct = (
                self.G(self.G_pred, output_size=input_size, random_affine=-rand_affine)
                if run_reconstruct
                else None
            )

        return self.G_pred, self.D_preds, self.reconstruct

    def train_g(self):
        """
        Train Generator
        :return:
        """
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        # Determine output size of G (dynamic change)
        output_size, random_affine = random_size(
            orig_size=self.input_tensor.shape[2:],
            curriculum=self.conf.curriculum,
            i=self.cur_epoch,
            epoch_for_max_range=self.conf.epoch_for_max_range,
            must_divide=self.conf.must_divide,
            min_scale=self.conf.min_scale,
            max_scale=self.conf.max_scale,
            max_transform_magnitude=self.conf.max_transform_magnitude,
        )

        # Add noise to G input for better generalization (make it ignore the 1/255 binning)
        self.input_tensor_noised = self.input_tensor

        if self.conf.G_noise:
            self.input_tensor_noised += (
                torch.rand_like(self.input_tensor) - 0.5
            ) * self.conf.G_noise

        # Generator forward pass
        self.G_pred = self.G(
            self.input_tensor_noised,
            output_size=output_size,
            random_affine=random_affine,
        )

        # Run generator result through discriminator forward pass
        self.scale_weights = get_scale_weights(
            i=self.cur_epoch,
            max_i=self.conf.D_scale_weights_epoch_for_even_scales,
            start_factor=self.conf.D_scale_weights_sigma,
            input_shape=self.G_pred.shape[2:],
            min_size=self.conf.D_min_input_size,
            num_scales_limit=self.conf.D_max_num_scales,
            scale_factor=self.conf.D_scale_factor,
        )
        d_pred_fake = self.D(self.G_pred, self.scale_weights)

        self.reconstruct = self.G(
            self.G_pred,
            output_size=self.input_tensor.shape[2:],
            random_affine=-random_affine,
        )
        self.loss_G_reconstruct = self.criterionReconstruction(
            self.reconstruct, self.input_tensor, self.loss_mask
        )

        # Calculate generator loss, based on discriminator prediction on generator result
        self.loss_G_GAN = self.criterionGAN(d_pred_fake, is_d_input_real=True)

        # Generator final loss
        # Weighted average of the two losses
        self.loss_G = (
            self.conf.reconstruct_loss_proportion * self.loss_G_reconstruct
            + self.loss_G_GAN
        )

        # Calculate gradients
        # Note that the gradients are propagated from the loss through discriminator and then through generator
        self.loss_G.backward()

        # Update weights
        # Note that only generator weights are updated (by definition of the G optimizer)
        self.optimizer_G.step()

        # Extra training for the inverse G. The difference between this and the reconstruction is the .detach() which
        # makes the training only for the inverse G and not for regular G.
        if self.cur_epoch > self.conf.G_extra_inverse_train_start_epoch:
            for _ in range(self.conf.G_extra_inverse_train):
                self.optimizer_G.zero_grad()
                self.inverse = self.G(
                    self.G_pred.detach(),
                    output_size=self.input_tensor.shape[2:],
                    random_affine=-random_affine,
                )
                self.loss_G_inverse = (
                    self.criterionReconstruction(
                        self.inverse, self.input_tensor, self.loss_mask
                    )
                    * self.conf.G_extra_inverse_train_ratio
                )
                self.loss_G_inverse.backward()
                self.optimizer_G.step()

    def train_d(self):
        """
        Train Discriminator
        :return:
        """
        # Zeroize gradients
        self.optimizer_D.zero_grad()

        real_example_with_noise = self.real_example
        if self.conf.D_noise:
            real_example_with_noise += (
                torch.rand_like(self.real_example[-1]) - 0.5
            ) * self.conf.D_noise

        # Discriminator forward pass over real example
        self.d_pred_real = self.D(real_example_with_noise, self.scale_weights)

        g_pred_with_noise = self.G_pred.detach()
        if self.conf.D_noise:
            g_pred_with_noise += (
                torch.rand_like(self.G_pred) - 0.5
            ) * self.conf.D_noise

        # Discriminator forward pass over generated example example
        self.d_pred_fake = self.D(g_pred_with_noise, self.scale_weights)

        # Calculate discriminator loss
        self.loss_D_fake = self.criterionGAN(self.d_pred_fake, is_d_input_real=False)
        self.loss_D_real = self.criterionGAN(self.d_pred_real, is_d_input_real=True)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        # Calculate gradients
        # Note that gradients are not propagating back through generator
        self.loss_D.backward()

        # Update weights
        # Note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

    def train_one_epoch(self, train_loader: DataLoader):
        """
        Train on epoch
        :param train_loader: Dataloader object
        :return:
        """
        for batch in tqdm(
            train_loader, dynamic_ncols=True, desc=f"Epoch {self.cur_epoch}"
        ):
            batch = batch.to(self.device)
            mask_crops = []
            mask_flag = False

            self.real_example = self.RandCrop(batch)
            self.input_tensor = self.real_example.to(self.device)
            self.loss_mask = torch.cat(mask_crops) if mask_flag else None

            # Run a single forward-backward pass on the model and update weights
            # One global epoch includes several epochs of generator and several of discriminator
            # (not necessarily equal)
            # noinspection PyRedeclaration
            for _ in range(self.conf.G_epochs):
                self.train_g()

            # noinspection PyRedeclaration
            for _ in range(self.conf.D_epocs):
                self.train_d()

        # Update learning rate scheduler
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()

        # Accumulate stats
        self.losses_G_gan[
            self.cur_epoch % self.conf.print_epoch_freq
        ] = self.loss_G_GAN.item()
        self.losses_D_fake[
            self.cur_epoch % self.conf.print_epoch_freq
        ] = self.loss_D_fake.item()
        self.losses_D_real[
            self.cur_epoch % self.conf.print_epoch_freq
        ] = self.loss_D_real.item()
        self.losses_G_reconstruct[
            self.cur_epoch % self.conf.print_epoch_freq
        ] = self.loss_G_reconstruct.item()

    @property
    def cur_epoch(self):
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, val):
        self._cur_epoch = val

    def save_checkpoint(self):
        """
        Save checkpoint of the model and optimizer on main process.
        """
        run_on_main(_save_checkpoint, kwargs={"trainer": self})


def _save_checkpoint(trainer: ScalerGANTrainer):
    """
    Save checkpoint function to run on main process.

    Args:
        trainer (ScalerGANTrainer): Trainer object
    """
    cur_epoch = trainer.cur_epoch
    multi_gpu = trainer.conf.distributed
    model_params_dict = {
        "G": trainer.G.module.state_dict() if multi_gpu else trainer.G.state_dict(),
        "D": trainer.D.module.state_dict() if multi_gpu else trainer.D.state_dict(),
        "optim_G": trainer.optimizer_G.state_dict(),
        "optim_D": trainer.optimizer_D.state_dict(),
        "sched_G": trainer.lr_scheduler_G.state_dict(),
        "sched_D": trainer.lr_scheduler_D.state_dict(),
        "loss": trainer.criterionGAN.state_dict(),
        "epoch": cur_epoch,
    }
    artifacts_dir = trainer.conf.artifacts_dir
    checkpoints_path = os.path.join(artifacts_dir, CHECKPOINT_DIR)
    os.makedirs(checkpoints_path, exist_ok=True)
    file_path = os.path.join(checkpoints_path, f"checkpoint_{cur_epoch:05}.pth.tar")
    logger.info(f"Saving checkpoint to '{file_path}'")
    torch.save(model_params_dict, file_path)
