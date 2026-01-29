# torch.rand(B, L, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified structure mimicking Wav2Vec2ForCTC's linear layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(16000, 512),  # Matches input length dimension
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.output = nn.Linear(256, 256)  # CTC head-like output
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.output(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Example input: batch_size=1, audio_length=16000 (common for 16kHz audio)
    return torch.rand(1, 16000, dtype=torch.float)

