# torch.rand(100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Compute rfft output (original model)
        output_rfft = torch.fft.rfft(x)
        
        # Compute stft output with default center=True (compared model)
        output_stft = torch.stft(
            x, 
            n_fft=100, 
            hop_length=101, 
            onesided=True, 
            return_complex=True, 
            center=True
        ).squeeze()
        
        # Return absolute difference between outputs
        return torch.abs(output_rfft - output_stft)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, dtype=torch.float32)

