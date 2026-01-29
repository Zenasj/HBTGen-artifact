# torch.rand(1, 257, 469, dtype=torch.complex64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, n_fft=512, hop_length=320, win_length=512, center=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length).cuda()  # Matches the issue's parameters
        self.center = center

    def forward(self, stft_tensor, length=150079):
        # Original ISTFT (problematic case)
        original_recon = torch.istft(
            stft_tensor,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            length=length
        )[0]  # Matches the example's indexing

        # Fixed ISTFT with padding logic (inspired by librosa's approach)
        start = self.n_fft // 2 if self.center else 0
        desired_total_length = start + length
        current_n_frames = stft_tensor.size(-1)
        expected_signal_length = self.n_fft + self.hop_length * (current_n_frames - 1)

        if expected_signal_length < desired_total_length:
            required_n_frames = (desired_total_length - self.n_fft) // self.hop_length + 1
            if (desired_total_length - self.n_fft) % self.hop_length != 0:
                required_n_frames += 1
            padding_needed = required_n_frames - current_n_frames
            if padding_needed > 0:
                pad = (0, padding_needed, 0, 0, 0, 0)  # Pad along the frames dimension
                padded_stft = F.pad(stft_tensor, pad, mode='constant', value=0)
            else:
                padded_stft = stft_tensor
        else:
            padded_stft = stft_tensor

        fixed_recon = torch.istft(
            padded_stft,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            length=length
        )[0]

        # Return whether the fixed ISTFT achieved the desired length
        return torch.tensor(fixed_recon.shape[0] == length, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 257, 469, dtype=torch.complex64)

