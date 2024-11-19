import glob
import os
import random
from typing import Optional
import torchaudio


from scaler_gan.scalergan_utils.scalergan_utils import (
    files_to_list,
    load_audio_to_np,
    mel_spectrogram,
    norm_audio_like_hifi_gan,
    sample_segment,
    crop_mel,
)
import torch


class MelDataset(torch.utils.data.Dataset):
    """
    MelDataset is data generator class that inherits torch.utils.data.Dataset class
    """

    def __init__(
        self,
        training_files: str,
        must_divide: float,
        segment_size: Optional[int] = 8192,
        n_fft: Optional[int] = 1024,
        num_mels: Optional[int] = 80,
        hop_size: Optional[int] = 256,
        win_size: Optional[int] = 1024,
        sampling_rate: Optional[int] = 22050,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = 8000,
        shuffle: Optional[bool] = True,
        n_cache_reuse: Optional[int] = 1,
        fmax_loss: Optional[float] = None,
        libri_flag:Optional[bool] = False,
        seed: Optional[int] = 1234,
    ):
        """
        Init
        :param training_files: The training files, can be directory with wav files or text file with list of wav files or libri files
        :param must_divide: Division factor
        :param segment_size: The size of the output segment
        :param n_fft: The number of FFT coefficients
        :param num_mels: The number of mel coefficients
        :param hop_size: The hop size in samples
        :param win_size: The frame (window) size in samples
        :param sampling_rate: The wav file sampling rate
        :param fmin: The lower frequency boundary
        :param fmax: The higher frequency boundary
        :param shuffle: Flag for shuffling or not
        :param n_cache_reuse: Number of item to reuse
        :param fmax_loss: The higher frequency boundary loss
        :param seed: Seed factor
        """
        # load list of files
        self.libri_train = False
        if(training_files == "libri_flag"):
            # Load list of files
            self.libri_train = True
            self.audio_files = self.load_libri_files(training_files)
            self.audio_files = sorted(self.audio_files)
        else:
            if os.path.isdir(training_files):
                self.audio_files = sorted(glob.glob(os.path.join(training_files, "*")))
            else:
                self.audio_files = files_to_list(training_files)

        if shuffle:
            # random.seed(seed)
            random.shuffle(self.audio_files)
        self.must_divide = must_divide
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get item for training
        :param index: Specific index in the dataset
        :return: mel spectrogram as tensor
        """
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio_to_np(filename,self.libri_train)
            audio = norm_audio_like_hifi_gan(audio)
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    "{} SR doesn't match target {} SR".format(
                        sampling_rate, self.sampling_rate
                    )
                )
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        audio = sample_segment(audio, self.segment_size)

        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax,
            center=False,
        )
        mel = crop_mel(mel, self.must_divide)
        return mel

    def __len__(self) -> int:
        """
        The size of the dataset
        :return: The total samples in the training dataset
        """
        return len(self.audio_files)

    def load_libri_files(self, training_files: str) -> list:
        dataset = torchaudio.datasets.LIBRISPEECH(root=training_files, url="train-clean-100", download=False)
        file_list = [item[0] for item in dataset]
        return file_list
