import argparse
from typing import Optional

import os
import json
import git

from scaler_gan.scalergan_utils.scalergan_utils import AttrDict, prepare_result_dir

class Config:
    """
    Config class that using cmd input arguments and parse them
    """

    def __init__(self):
        """
        Init
        """
        self.parser = argparse.ArgumentParser()
        self.conf = None
        wt_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        # Paths
        self.parser.add_argument('-f', '--input_file', default=os.path.join(wt_dir, 'data/input.txt'), help='path to inputs filenames path')
        self.parser.add_argument('-inf', '--inference_file', default=os.path.join(wt_dir, 'data/inference.txt'), help='path to test filenames path')
        self.parser.add_argument('-o', '--output_dir', default='output/trained_models', help='path to a directory to save results to')
        self.parser.add_argument('--artifacts_dir', default=None, help='path to a directory to save artifacts results to. If not given, will be saved in output_dir')
        self.parser.add_argument('-n', '--name', default='scaler_gan', help='name of current experiment, to be used for saving the results')
        self.parser.add_argument('--resume', default=None, help='checkpoint to resume from')
        self.parser.add_argument('--fine_tune', action='store_true', help='fine tune the model from a given checkpoint specified in "--checkpoint_path"')
        self.parser.add_argument('-mc', '--mel_config', type=str, default=os.path.join(wt_dir, 'scaler_gan/configs/mel_config.json'), help='JSON file for mel configuration')
        self.parser.add_argument('-d', '--debug', action='store_true', help="Debug mode, wont save results")

        # Inference
        self.parser.add_argument('-cp', '--checkpoint_path', type=str, default=os.path.join(wt_dir, 'pretrained_models/lj_speech_model.pth.tar'), help='checkpoint path for inference')
        self.parser.add_argument("--infer_scales", type=float, default=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5], nargs="+", help="list of scales for outputs")
        self.parser.add_argument('--infer_plt', action="store_true",  help='plot the mels inference results')
        self.parser.add_argument('--infer_hifi', action="store_true", help='use hifi-gan inference to generate audio from mels')
        self.parser.add_argument('--hifi_checkpoint', type=str, default=os.path.join(wt_dir,"pretrained_models/hifi_checkpoint_v1"), help='hifi gan checkpoint path for inference')
        self.parser.add_argument('--hifi_config', type=str, default=os.path.join(wt_dir, "scaler_gan/configs/hifi_config.json"), help='hifi gan config path for inference')
        


        # Architecture (Generator)
        self.parser.add_argument('--G_base_channels', type=int, default=64, help='# of base channels in G')
        self.parser.add_argument('--G_num_resblocks', type=int, default=6, help='# of resblocks in G\'s bottleneck')
        self.parser.add_argument('--G_num_downscales', type=int, default=3, help='# of downscaling layers in G')
        self.parser.add_argument('--G_use_bias', type=bool, default=True, help='Determinhes whether bias is used in G\'s conv layers')
        self.parser.add_argument('--G_skip', type=bool, default=True, help='Determines wether G uses skip connections (U-net)')
        self.parser.add_argument('--G_noise', type=float, default=None, help='Determines how much does G addes noise to input')


        # Architecture (Discriminator)
        self.parser.add_argument('--D_base_channels', type=int, default=64, help='# of base channels in D')
        self.parser.add_argument('--D_max_num_scales', type=int, default=5, help='Limits the # of scales for the multiscale D')
        self.parser.add_argument('--D_scale_factor', type=float, default=1.2, help='Determines the downscaling factor for multiscale D')
        self.parser.add_argument('--D_scale_weights_sigma', type=float, default=1.4, help='Determines the downscaling factor for multiscale D')
        self.parser.add_argument('--D_min_input_size', type=int, default=13, help='Determines the downscaling factor for multiscale D')
        self.parser.add_argument('--D_scale_weights_epoch_for_even_scales', type=int, default=25000, help='Determines the downscaling factor for multiscale D')
        self.parser.add_argument('--D_noise', type=float, default=None, help='Determines how much does D addes noise to input')


        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=0.00005, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=0.00005, help='initial learning rate for discriminator')
        self.parser.add_argument('--lr_start_decay_epoch', type=float, default=None, help='epoch from which linear decay of lr starts until max_epoch')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--curriculum', type=bool, default=True, help='Enable curriculum learning')
        self.parser.add_argument('--epoch_for_max_range', type=int, default=200, help='In curriculum learning, when getting to this epoch all range is covered')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=256, help='input is cropped to this size')
        self.parser.add_argument('--output_crop_size', type=int, default=256, help='output is cropped to this size')
        self.parser.add_argument('--max_scale', type=float, default=1.8, help='max retargeting scale')
        self.parser.add_argument('--min_scale', type=float, default=0.3, help='min retargeting scale')
        self.parser.add_argument('--must_divide', type=int, default=8, help='must divide parameter for consistent output size')
        self.parser.add_argument('--max_transform_magnitude', type=float, default=0.0, help='max manitude of geometric transformation')

        # GPU & CPU settings
        self.parser.add_argument("--device", type=str, default='cpu', help="which device to use cpu or cuda")
        self.parser.add_argument("--batch_size", type=int, default=24, help="batch size")
        self.parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")
        
        # DDP
        self.parser.add_argument("--distributed", action='store_true', help="Run with distributed data parallel")
        self.parser.add_argument("--distributed_backend", type=str, default="nccl", help="Which distributed backend to use")
        self.parser.add_argument("--find_unused_parameters", action='store_true')
        self.parser.add_argument("--local_rank",type=int, default=0)

        # Monitoring display frequencies
        self.parser.add_argument('--print_epoch_freq', type=int, default=1, help='epochs frequency of showing training results on console')
        self.parser.add_argument('--save_snapshot_freq', type=int, default=1, help='frequency of saving the latest results')
        self.parser.add_argument('--reconsturct_plot_freq', type=int, default=5, help='save reconstruct and origin mel picture each # epochs')
        self.parser.add_argument('--G_pred_freq_plot', type=int, default=5, help='save G pred and origin mel picture each # epochs')
        self.parser.add_argument('--wandb', action='store_true',  help='monitor data to weights & biases site')

        # epochs
        self.parser.add_argument('--num_epochs', type=int, default=500, help='max # of epochs')
        self.parser.add_argument('--G_epochs', type=int, default=1, help='# of sub-epochs for the generator per each global epoch')
        self.parser.add_argument('--D_epocs', type=int, default=1, help='# of sub-epochs for the discriminator per each global epoch')

        # Losses
        self.parser.add_argument('--reconstruct_loss_proportion', type=float, default=0.1, help='relative part of reconstruct-loss (out of 1)')
        self.parser.add_argument('--reconstruct_loss_stop_epoch', type=int, default=2000, help='from this epoch and on, reconstruct loss is deactivated')
        self.parser.add_argument('--G_extra_inverse_train', type=int, default=1, help='number of extra training epochs for G on inverse direction')
        self.parser.add_argument('--G_extra_inverse_train_start_epoch', type=int, default=200, help='number of extra training epochs for G on inverse direction')
        self.parser.add_argument('--G_extra_inverse_train_ratio', type=int, default=1.0, help='number of extra training epochs for G on inverse direction')
        self.parser.add_argument('--use_L1', type=bool, default=True, help='Determine whether to use L1 or L2 for reconstruction')

        self.parser.add_argument('-v', '--verbose', action='store_true', help='if specified, print more debugging information')

    def parse(self, inference_mode: Optional[bool] = False) -> argparse.Namespace:
        """
        Parse the cmd input arguments
        :param inference_mode: inference mode parsing, if true, wont create a new directory and copy code
        :return: The argument parser object as an argparse.Namespace
        """
        # Parse arguments
        self.conf = self.parser.parse_args()

        # set W&B flag
        if not self.conf.wandb or self.conf.debug:
            os.environ["WANDB_MODE"] = "offline"  # run offline

        with open(self.conf.mel_config) as f:
            data = f.read()
        mel_prms_json = json.loads(data)
        mel_params = AttrDict(mel_prms_json)
        setattr(self.conf, "mel_params", mel_params)
        # Create results dir if does not exist
        self.conf.output_dir = prepare_result_dir(self.conf, self.conf.debug, inference_mode)
        return self.conf
