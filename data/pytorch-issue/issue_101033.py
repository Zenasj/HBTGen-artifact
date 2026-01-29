# torch.rand(100, 1, 15, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, tf_n_channels=3, device=torch.device('cpu')):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=tf_n_channels * 5, nhead=tf_n_channels, dim_feedforward=60,
                                                   dropout=0.0, device=device, dtype=torch.float32)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return self.transformer(_input, is_causal=True, mask=torch.ones((_input.size(0), _input.size(0)),
                                                                        dtype=torch.bool,
                                                                        device=_input.device).triu(diagonal=1))

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MyModel(device=device)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.randn((100, 1, 15), device=device, dtype=torch.float32)

