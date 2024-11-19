import logging
import math
import os
import sys
from argparse import Namespace
from random import randint, random
from shutil import copy
from time import localtime, strftime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as f
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from matplotlib import pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm

import wandb
from scaler_gan.distributed.distributed import run_on_main
from scaler_gan.scalergan_utils.global_logger import logger

MAX_WAV_VALUE = 32768.0
RECONSTURCT_PLT = "reconstruct_mel_plt"
G_PRED_DIR = "g_pred"
MELS_IMGS_DIR = "mels_images"
RECONSTRUCT = "reconstruct"
G_PRED_PLT = "g_pred_plt"
CROPPED_DIR = "cropped_audios"

mel_basis = {}
hann_window = {}


class AttrDict(dict):
    """
    A dictionary that allows for dot notation access
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def crop_mel(mel: torch.Tensor, must_divide: float) -> torch.Tensor:
    """
    Croping the mel spectrogram
    :param mel: Mel spectrogram
    :param must_divide: division factor
    :return: cropped mel spectrogram
    """
    input_mel_cropped = mel[
        :,
        : (mel.shape[1] // must_divide) * must_divide,
        : (mel.shape[2] // must_divide) * must_divide,
    ]
    return input_mel_cropped


def random_size(
    orig_size: List[int],
    curriculum: Optional[bool] = True,
    i: Optional[float] = None,
    epoch_for_max_range: Optional[int] = None,
    must_divide: Optional[float] = 8.0,
    min_scale: Optional[float] = 0.25,
    max_scale: Optional[float] = 2.0,
    max_transform_magnitude: Optional[float] = 0.3,
) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Get random size
    :param orig_size: The original size
    :param curriculum: Use curriculum or not
    :param i: epoch index
    :param epoch_for_max_range:  epoch for max range
    :param must_divide: division factor
    :param min_scale: Min scale factor
    :param max_scale: Max scale factor
    :param max_transform_magnitude: max transform magnitude factor
    :return: Tuple with new size tuple and random affine
    """
    cur_max_scale = (
        1.0 + (max_scale - 1.0) * np.clip(1.0 * i / epoch_for_max_range, 0, 1)
        if curriculum
        else max_scale
    )
    cur_min_scale = (
        1.0 + (min_scale - 1.0) * np.clip(1.0 * i / epoch_for_max_range, 0, 1)
        if curriculum
        else min_scale
    )
    cur_max_transform_magnitude = (
        max_transform_magnitude * np.clip(1.0 * i / epoch_for_max_range, 0, 1)
        if curriculum
        else max_transform_magnitude
    )

    # set random transformation magnitude. scalar = affine, pair = homography.
    random_affine = (
        -cur_max_transform_magnitude
        + 2 * cur_max_transform_magnitude * np.random.rand(2)
    )

    # set new size for the output image
    new_size = np.array(orig_size) * [
        1,
        cur_min_scale + (cur_max_scale - cur_min_scale) * random(),
    ]

    return (
        tuple(np.uint32(np.ceil(new_size * 1.0 / must_divide) * must_divide)),
        random_affine,
    )


def get_scale_weights(
    i: int,
    max_i: int,
    start_factor: float,
    input_shape: torch.tensor,
    min_size: int,
    num_scales_limit: int,
    scale_factor: float,
) -> np.ndarray:
    """
    get scale weights for the current epoch index i
    Args:
        i (int): epoch index
        max_i (int): max epoch index, used for linear scaling.
        start_factor (float): start factor for the scale weights
        input_shape (torch.tensor): input shape
        min_size (int): min size for the scale weights  
        num_scales_limit (int): max number of scales
        scale_factor (float): scale factor for the scale weights

    Returns:
        np.ndarray: scaled weights
    """
    num_scales = np.min(
        [
            np.int(
                np.ceil(
                    np.log(np.min(input_shape) * 1.0 / min_size) / np.log(scale_factor)
                )
            ),
            num_scales_limit,
        ]
    )
    factor = start_factor ** ((max_i - i) * 1.0 / max_i)

    un_normed_weights = factor ** np.arange(num_scales)
    weights = un_normed_weights / np.sum(un_normed_weights)

    return weights


def prepare_result_dir(
    conf: Namespace,
    debug_mode: Optional[bool] = False,
    inference_mode: Optional[bool] = False,
) -> str:
    """
    Prepare the result directory
    :param conf: Configuration
    :param debug_mode: Debug mode, default is False
    :param inference_mode: Inference mode, default is False
    :return: updated output_dir
    """
    # Create results directory
    if debug_mode:
        conf.output_dir = os.path.join(conf.output_dir, "debug")
    else:
        if inference_mode:
            conf.name = f"inference_{conf.name}"
        conf.output_dir = os.path.join(
            conf.output_dir, conf.name + strftime("_%b_%d_%H_%M_%S", localtime())
        )
    os.makedirs(conf.output_dir, exist_ok=True)
    if not conf.artifacts_dir:
        conf.artifacts_dir = os.path.join(conf.output_dir, "artifacts")
    os.makedirs(conf.artifacts_dir, exist_ok=True)
    # Put a copy of all *.py files in results path, to be able to reproduce experimental results
    if not inference_mode:
        if conf.resume:
            run_on_main(
                main_proc_copy,
                kwargs={
                    "checkpoint_to_copy": conf.resume,
                    "output_dir": conf.output_dir,
                },
            )
    return conf.output_dir


def main_proc_copy(checkpoint_to_copy: str, output_dir: str):
    """
    Copy the checkpoint to the output directory only by main process.

    Args:
        checkpoint_to_copy (str): checkpoint to copy
        copy_path (str): copy path
    """
    copy_path = os.path.join(output_dir, "starting_checkpoint.pth.tar")
    copy(checkpoint_to_copy, copy_path)


def homography_based_on_top_corners_x_shift(rand_h):
    p = np.array(
        [
            [
                1.0,
                1.0,
                -1,
                0,
                0,
                0,
                -(-1.0 + rand_h[0]),
                -(-1.0 + rand_h[0]),
                -1.0 + rand_h[0],
            ],
            [0, 0, 0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, -1, 0, 0, 0, 1 + rand_h[1], 1 + rand_h[1], 1 + rand_h[1]],
            [0, 0, 0, -1, -1, -1, 1, 1, 1],
            [1, 0, -1, 0, 0, 0, 1, 0, -1],
            [0, 0, 0, 1, 0, -1, 0, 0, 0],
            [-1, 0, -1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, -1, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    b = np.zeros((9, 1), dtype=np.float32)
    b[8, 0] = 1.0
    h = np.dot(np.linalg.inv(p), b)
    return torch.from_numpy(h).view(3, 3)


def homography_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of homography matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Tensor): input batch of homography matrices (:math:`N \times 3 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    """
    a = 1
    b = 1
    y, x = torch.meshgrid(
        [
            torch.linspace(-b, b, np.int(size[-2] * a)),
            torch.linspace(-b, b, np.int(size[-1] * a)),
        ],
        indexing="ij",
    )
    n = np.int(size[-2] * a) * np.int(size[-1] * a)
    hxy = torch.ones(n, 3, dtype=torch.float).to(theta.device)
    hxy[:, 0] = x.contiguous().view(-1)
    hxy[:, 1] = y.contiguous().view(-1)
    out = hxy[None, ...].matmul(theta.transpose(1, 2))
    # normalize
    out = out[:, :, :2] / out[:, :, 2:]
    return out.view(theta.shape[0], np.int(size[-2] * a), np.int(size[-1] * a), 2)


def save_mels_plt(
    name_2_np_dict: Dict[str, np.ndarray], plt_path: str, plt_wandb_name: str
):
    """
    save a plot of the given mels

    Args:
        name_2_np_dict (Dict[str, np.ndarray]): Dictionary of names to numpy arrays
        plt_path (str): Path to save the plot
        plt_wandb_name (str): Name of the plot in wandb
    """
    os.makedirs(os.path.dirname(plt_path), exist_ok=True)

    n_plts = len(name_2_np_dict)
    fig = plt.figure(figsize=(4 * n_plts, 2))
    rows = math.ceil(n_plts / 2)
    for i, (title, input) in enumerate(name_2_np_dict.items()):
        sub_plot = fig.add_subplot(rows, 2, i + 1)
        sub_plot.title.set_text(title)
        vec = input.cpu().data.numpy() if torch.is_tensor(input) else input
        plt.imshow(
            vec,
            origin="lower",
            interpolation="none",
            aspect="auto",
            extent=(0, input.shape[1], 0, input.shape[0]),
        )
    plt.savefig(plt_path)
    save_wandb_plt(plt_wandb_name)
    plt.close(fig)


def save_wandb_plt(plot_name: str):
    """
    Save a plot to wandb
    Args:
        plot_name (str): Name of the plot in wandb
    """
    wandb.log({plot_name: wandb.Image(plt)}, commit=False)


def create_mel_from_audio(
    wav: torch.tensor, mels_params: Dict[str, Any], must_divide: int
) -> torch.tensor:
    """
    Create a mel spectrogram from an audio tensor
    
    Args:
        wav (torch.tensor): Audio tensor
        mels_params (Dict[str, Any]): Mel spectrogram parameters
        must_divide (int): Must divide int, divide the audio to fit the model

    Returns:
        torch.tensor: Mel spectrogram
    """
    # crop_audio_by_must_divide(wav, mels_params.hop_size, must_divide)
    mel = mel_spectrogram(
        wav,
        mels_params["n_fft"],
        mels_params["num_mels"],
        mels_params["sampling_rate"],
        mels_params["hop_size"],
        mels_params["win_size"],
        mels_params["fmin"],
        mels_params["fmax"],
        center=False,
    )
    return mel.unsqueeze(0)


def crop_audio_by_must_divide(
    wav: torch.tensor, hop_size: int, must_divide: int
) -> torch.tensor:
    """
    Crop the audio to be divided by must divide
    Args:
        wav (torch.tensor): Audio tensor
        hop_size (int): Hop size
        must_divide (int): Must divide int, divide the audio to fit the model

    Returns:
        torch.tensor: cropped audio tensor
    """

    frames_per_audio = wav.size // hop_size
    frames_per_audio_divded = frames_per_audio - (frames_per_audio % must_divide)
    wav = wav[: frames_per_audio_divded * hop_size]
    return wav


def load_and_norm_audio(audio_path: str, model_sampling_rate: int) -> torch.tensor:
    """
    Load audio, normalize it and convert it to tensor
    Args:
        audio_path (str): Path to the audio file
        model_sampling_rate (int): Sampling rate of the model

    Raises:
        ValueError: If the audio sampling rate is not the same as the model sampling rate

    Returns:
        torch.tensor: Normalized audio tensor.
    """
    audio, sampling_rate = load_audio_to_np(audio_path)
    audio = norm_audio_like_hifi_gan(audio)
    if sampling_rate != model_sampling_rate:
        raise ValueError(
            "{} SR doesn't match target {} SR".format(
                sampling_rate, model_sampling_rate
            )
        )
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    return audio


def save_cropped_audios(
    audio_files: List[str], artifacts_dir: str, hop_size: int, must_divide: int
) -> List[str]:
    """
    Save cropped audios to fit the model, in the artifacts dir.

    Args:
        audio_files (List[str]): List of audio files to crop
        artifacts_dir (str): Path to artifacts dir
        hop_size (int): Hop size for cropping
        must_divide (int): Must divide for cropping
    Returns:
        List[str]: List of cropped audio files
    """
    global logger
    cropped_lst = []

    logger.info(f"Saving cropped audios to '{artifacts_dir}'")
    for audio_file in tqdm(audio_files, unit="example"):
        file_name_no_ext = os.path.splitext(os.path.basename(audio_file))[0]
        audio, sr = load_audio_to_np(audio_file)
        # audio, sr = load_audio_to_torch(audio_file)
        audio = crop_audio_by_must_divide(audio, hop_size, must_divide)
        save_path = os.path.join(artifacts_dir, f"{file_name_no_ext}_cropped.wav")
        cropped_lst.append(save_path)
        wavfile.write(save_path, sr, audio)
    return cropped_lst


def load_audio_to_np(full_path: str,libri_flag: bool) -> Tuple[np.ndarray, int]:
    """
    Load wav file
    :param full_path: Path to the wav file
    :return: Tuple with the raw data of the wav file and the sampling rate
    """
    if(libri_flag):
        waveform, sampling_rate = torchaudio.load(full_path)
        data = waveform.numpy()
        if data.ndim > 1:
            data = data[0]
    else:
        sampling_rate, data = wavfile.read(full_path)
    return data, sampling_rate


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
    center: Optional[bool] = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram
    :param y: The input signal the need to be transformed to mel spectrogram
    :param n_fft: The number of FFT coefficients
    :param num_mels: The number of mel coefficients
    :param sampling_rate: The signal sampling rate
    :param hop_size: The hop size in samples
    :param win_size: The frame (window) size in samples
    :param fmin: The lower frequency boundary
    :param fmax: The higher frequency boundary
    :param center: Flag for centering the fourier transform or not default is False
    :return: Normalized mel spectrogram
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def dynamic_range_compression_torch(
    x: torch.Tensor, c: Optional[int] = 1, clip_val: Optional[float] = 1e-5
) -> torch.Tensor:
    """
    Calculate the dynamic range compression
    :param x: Input tensor
    :param c: Multiply factor
    :param clip_val: Clipping factor
    :return: The dynamic range compression as tensor
    """
    return torch.log(torch.clamp(x, min=clip_val) * c)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """
    Normalize the spectral magnitudes
    :param magnitudes: Magnitudes values
    :return: Normalized spectral
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output


def sample_segment(
    audio: torch.Tensor, n_samples: int, ret_idx: Optional[bool] = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
    """
    Samples a random segment of `n_samples` from `audio`.
    if audio is shorter than `n_samples` then the original audio is zero padded.
    :param audio: Audio tensor of shape [1, T]
    :param n_samples: The output number of samples of the new segment
    :param ret_idx: if True then the start and end indices will be returned
    :return: The new segment and if ret_idx is set to True the start and end samples are also return
    """
    start, end = 0, audio.shape[1]
    diff = abs(audio.shape[1] - n_samples)

    if audio.shape[1] > n_samples:
        start = randint(0, diff)
        end = start + n_samples
        audio = audio[:, start:end]

    elif audio.shape[1] < n_samples:
        audio = f.pad(audio, (0, diff))

    if ret_idx:
        return audio, (start, end)
    return audio


def files_to_list(filename: str) -> List[str]:
    """
    Takes a text file of filenames and makes a list of filenames
    :param filename: Path to the text file
    :return: List of file paths
    """

    with open(filename) as f:
        files = [line.rstrip() for line in f]

    return files


def load_audio_to_torch(full_path: str) -> Tuple[torch.Tensor, int]:
    """
    Loads audio data into torch array
    :param full_path: Path to the wav file
    :return: Tuple with the raw data of the wav file and the sampling rate
    """
    audio, sampling_rate = torchaudio.load(full_path)

    return audio, sampling_rate


def init_logger(log_file: str = "log.txt", log_level: str = "INFO") -> None:
    """
    Initialize the logger
    :param log_file: Path to the log file
    :param log_level: The log level
    """
    global logger
    logger.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch_formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    fh_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]:  %(message)s"
    )

    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)  # or any level you want
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    logging.getLogger("PIL").setLevel(logging.WARNING)


def save_g_pred_plot(
    input_mel: torch.Tensor, g_pred: torch.Tensor, plt_path: str
) -> None:
    """
    Save a plot of the input mel spectrogram and the generated mel spectrogram
    Args:
        input_mel (torch.Tensor): input mel spectrogram
        g_pred (torch.Tensor): generated mel spectrogram
        plt_path (str): path to save the plot
    """
    mels_dict = {
        "mel": input_mel,
        "g_pred": g_pred,
    }
    save_mels_plt(mels_dict, plt_path, G_PRED_PLT)


def save_reconstruct_plot(
    input_mel: torch.Tensor, reconstruct_mel: torch.Tensor, plt_path: str
):
    """
    Save a plot of the input mel spectrogram, the reconstructed mel spectrogram
    and the difference between them.

    Args:
        input_mel (torch.Tensor): input mel spectrogram
        reconstruct_mel (torch.Tensor): reconstructed mel spectrogram
        plt_path (str): path to save the plot
    """
    difference = input_mel - reconstruct_mel
    mels_dict = {
        "origin mel": input_mel,
        "reconstruct": reconstruct_mel,
        "difference": difference,
    }
    save_mels_plt(mels_dict, plt_path, RECONSTURCT_PLT)


def norm_audio_like_hifi_gan(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio like HiFi-GAN
    :param audio: Audio tensor
    :return: Normalized audio tensor
    """
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    return audio
