import torch
import torchaudio
import torchaudio.functional as F

# torch.rand(1, 16000, dtype=torch.float32)  # Waveform input shape (batch, samples)
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16000, 100 * 30)  # Arbitrary output shape (1, 100, 30)
    
    def forward(self, x):
        x = self.fc(x)
        return x.view(1, 100, 30)  # Emission shape (batch, time, classes)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 16000, dtype=torch.float32)

