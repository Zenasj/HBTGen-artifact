from typing import Optional

import torch
import torchaudio

import numpy as np

from tests.__init__ import (
    __target_clock__ as TARGET_CLOCK,
    __number_of_test_data_vals__ as NUMBER_OF_TEST_DATA_VALS,
)

# Set general parameters:
TARGET_DEVICE = "CUDA"
TARGET_FREQUENCY: int = 440
NUMBER_OF_FFT_SLOTS: int = 1024
HOP_LENGTH: Optional[int] = None
NUMBER_OF_MEL_SLOTS: int = 128

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    target_device = torch.device(
        "cuda" if (TARGET_DEVICE == "CUDA" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_default_dtype(d=torch.float64)
    # torch.set_default_device(device=target_device)
    print(f"Using device {target_device}")
    sampling_vec: np.ndarray = np.arange(NUMBER_OF_TEST_DATA_VALS) / TARGET_CLOCK
    frequency_vec: np.ndarray = np.sin(
        2 * np.pi * TARGET_FREQUENCY * sampling_vec
    ).astype("float64")
    frequency_tensor: torch.Tensor = torch.Tensor(frequency_vec).to(target_device)
    print(f"Tensor is: {frequency_tensor}")
    mel_spectrogram: torch.Tensor = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_CLOCK,
        n_fft=NUMBER_OF_FFT_SLOTS,
        hop_length=HOP_LENGTH,
        n_mels=NUMBER_OF_MEL_SLOTS,
    )(frequency_tensor)
    print(f"Obtained MEL-Spectrogram: {mel_spectrogram}")