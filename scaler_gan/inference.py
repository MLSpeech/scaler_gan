import os
import sys

import git

wt_dir = git.Repo(".", search_parent_directories=True).working_tree_dir
hifi_abs_dir = os.path.join(wt_dir, "hifi_gan")
try:
    sys.path.index(wt_dir)
except ValueError:
    sys.path.append(wt_dir)
try:
    sys.path.index(hifi_abs_dir)
except ValueError:
    sys.path.append(hifi_abs_dir)

import json
from argparse import Namespace

import matplotlib
import numpy as np
import torch
import tqdm
from tqdm import tqdm

import hifi_gan.inference_e2e as inference_e2e
import wandb
from hifi_gan.env import AttrDict
from scaler_gan.configs.configs import Config
from scaler_gan.scalergan_utils.global_logger import logger
from scaler_gan.scalergan_utils.scalergan_utils import (CROPPED_DIR, G_PRED_DIR,
                                                       MELS_IMGS_DIR,
                                                       create_mel_from_audio,
                                                       files_to_list,
                                                       init_logger,
                                                       load_and_norm_audio,
                                                       save_cropped_audios,
                                                       save_g_pred_plot)
from scaler_gan.trainer.scalerGAN import ScalerGANTrainer

matplotlib.use("agg")
MELS_NPY = "mels_npy"
INFERENCE_LOG = "inference.log"
WAVS = "wavs"


def generate_scaled_mel(
    gan, input_tensor, time_scale, must_divide, affine=None, size_instead_scale=False,
):
    freq_scale = 1
    with torch.no_grad():
        in_size = input_tensor.shape[2:]
        if size_instead_scale:
            out_size = time_scale
        else:
            out_size = (
                np.uint32(
                    np.floor(freq_scale * in_size[0] * 1.0 / must_divide) * must_divide
                ),
                np.uint32(
                    np.floor(time_scale * in_size[1] * 1.0 / must_divide) * must_divide
                ),
            )
        output_tensor, _, _ = gan.inference(
            input_tensor=input_tensor,
            input_size=in_size,
            output_size=out_size,
            rand_affine=affine,
            run_d_pred=False,
            run_reconstruct=False,
        )
        return output_tensor.squeeze(0)


def inference(gan: ScalerGANTrainer):
    """
    Inference function for ScalerGAN model on a given audio files.
    Args:
        gan (ScalerGANTrainer): ScalerGANTrainer object
    """
    conf = gan.conf
    output_mel_dir = os.path.join(conf.artifacts_dir, MELS_NPY)
    audio_cropped_dir = os.path.join(conf.artifacts_dir, CROPPED_DIR)
    output_mel_plt_dir = os.path.join(conf.artifacts_dir, MELS_IMGS_DIR, G_PRED_DIR)
    output_wav_dir = os.path.join(conf.artifacts_dir, WAVS)

    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(audio_cropped_dir, exist_ok=True)
    os.makedirs(output_mel_plt_dir, exist_ok=True)

    logger.info(f"Artifacts path: '{conf.artifacts_dir}'")
    logger.info(f"Saved predicted mels to: '{output_mel_dir}'")
    if conf.infer_plt:
        logger.info(f"Saved predicted mels plots to: '{output_mel_plt_dir}'")

    audio_files = files_to_list(conf.inference_file)

    cropped_audio_files = save_cropped_audios(
        audio_files, audio_cropped_dir, conf.mel_params.hop_size, conf.must_divide
    )

    for scale in conf.infer_scales:
        logger.info(f"Predicting scale: {scale}")
        for audio_file in tqdm(cropped_audio_files, unit="example"):
            inference_one_mel(
                audio_file, scale, gan, conf, output_mel_dir, output_mel_plt_dir
            )

    if conf.infer_hifi:
        hifi_inference(
            conf.hifi_checkpoint, conf.hifi_config, output_mel_dir, output_wav_dir
        )


def inference_one_mel(
    audio_file: str,
    scale: int,
    gan: ScalerGANTrainer,
    conf: Namespace,
    output_mel_dir: str,
    output_mel_plt_dir: str,
):
    """
    Inference function for ScalerGAN model on a given audio file.
    Args:
        audio_file (str): Path to audio file
        scale (int): Scale to predict
        gan (ScalerGANTrainer): ScalerGANTrainer object
        conf (Namespace): Configuration object
        output_mel_dir (str): Path to directory to save predicted mels
        output_mel_plt_dir (str): Path to directory to save predicted mels plot
    """
    file_name_no_ext = os.path.splitext(os.path.basename(audio_file))[0]
    infer_filename = f"{file_name_no_ext}_{scale}"
    file_numpy_path = os.path.join(output_mel_dir, infer_filename)
    audio = load_and_norm_audio(audio_file, conf.mel_params["sampling_rate"])
    input_mel = create_mel_from_audio(audio, conf.mel_params, conf.must_divide)
    inference_mel = generate_scaled_mel(gan, input_mel, scale, conf.must_divide)

    np.save(file_numpy_path, inference_mel.cpu().data.numpy())

    if conf.infer_plt:
        plot_inference(input_mel, inference_mel, infer_filename, output_mel_plt_dir)


def plot_inference(
    input_mel: torch.tensor,
    inference_mel: torch.tensor,
    infer_filename: str,
    output_mel_plt_dir: str,
):
    """
    Plot inference function for ScalerGAN model on a given audio file.
    Args:
        input_mel (torch.tensor): Input mel
        output_mel (torch.tensor): Output mel
        infer_filename (str): Name of inference file
        output_mel_plt_dir (str): Path to directory to save predicted mels plot
    """
    os.makedirs(output_mel_plt_dir, exist_ok=True)
    plt_path = os.path.join(output_mel_plt_dir, f"{infer_filename}.jpg")
    save_g_pred_plot(input_mel.squeeze(), inference_mel.squeeze(), plt_path)


def hifi_inference(
    hifi_checkpoint: str, hifi_config: str, output_mel_dir: str, output_wav_dir: str
):
    """
    Inference function for HiFi-GAN model on a given mel files.
    
    Args:
        hifi_checkpoint (str): Path to hifi checkpoint
        hifi_config (str): Path to hifi config
        output_mel_dir (str): Path to directory of the predicted mels as hifi input
        output_wav_dir (str): Path to directory to save predicted wavs
    """
    hifi_e2e_dict = {
        "input_mels_dir": os.path.abspath(output_mel_dir),
        "output_dir": os.path.abspath(output_wav_dir),
        "checkpoint_file": os.path.abspath(hifi_checkpoint),
    }
    hifi_e2e_conf = Namespace(**hifi_e2e_dict)
    with open(hifi_config) as f:
        data = f.read()
    json_config = json.loads(data)
    inference_e2e.h = AttrDict(json_config)
    inference_e2e.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running HiFi-GAN inference and saving to: '{output_wav_dir}'")
    inference_e2e.inference(hifi_e2e_conf)


def main():
    conf = Config().parse(inference_mode=True)
    project_name = os.path.basename(conf.output_dir)
    wandb.init(
        project="ScalerGAN", entity="ScalerGAN", name=project_name, config=vars(conf)
    )

    gan = ScalerGANTrainer(conf, inference=True)
    log_level = "DEBUG" if conf.verbose or conf.debug else "INFO"
    log_file = os.path.join(conf.artifacts_dir, INFERENCE_LOG)
    init_logger(log_file=log_file, log_level=log_level)
    inference(gan)
    logger.info(f"Done predictions, Artifacts path: '{conf.artifacts_dir}'")


if __name__ == "__main__":
    main()
