# torch.rand(100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 100
        self.hop_length = 25
        self.win_length = 5
        self.return_complex = True
        self.onesided = False

    def forward(self, x):
        # Compute STFT with center=True and center=False
        stft_center_true = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            return_complex=self.return_complex,
            onesided=self.onesided
        )
        stft_center_false = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            return_complex=self.return_complex,
            onesided=self.onesided
        )
        return stft_center_true, stft_center_false

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the test case in the issue (1D tensor of length 100)
    return torch.rand(100, dtype=torch.float32)

