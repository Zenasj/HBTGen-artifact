# torch.rand(B, H, dtype=torch.float32)  # Input shape is (batch, audio_length)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimic Wav2Vec2ForCTC structure where 'conv' is part of feature_extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.encoder = nn.Linear(512, 256)  # Simplified encoder layer

    def forward(self, x):
        # Input x is (batch, audio_length)
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.feature_extractor(x)
        x = self.encoder(x.mean(dim=-1))  # Simplified pooling and encoding
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Random audio input of shape (batch, audio_length)
    return torch.rand(1, 16000)  # Example input length of 16kHz 1-second audio

