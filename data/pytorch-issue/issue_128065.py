# torch.rand(B, 1, T, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate  # Required for STFT parameters
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Simulate compress and decompress flow
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def my_model_function():
    # Initialize with 44.1kHz sample rate (matches model_type="44khz" in the issue)
    return MyModel(sample_rate=44100)

def GetInput():
    # Random audio tensor with shape [batch, channels, time_samples]
    # Matches modified signal (1 channel) and typical audio frame length
    return torch.rand(1, 1, 1024, dtype=torch.float32)

