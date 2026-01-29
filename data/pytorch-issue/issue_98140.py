# torch.rand(1, 10, 576, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Transformer with d_model=576 (problematic size causing is_sm80 error on non-A100 GPUs)
        self.transformer = nn.Transformer(
            d_model=576,  # Key parameter triggering the error on unsupported hardware
            nhead=8,      # Must divide d_model (576 / 8 = 72)
            num_encoder_layers=1,
            num_decoder_layers=1,
            batch_first=False  # Matches the input format in reported examples
        )

    def forward(self, x):
        # Mimics the reported use case where src and tgt are identical inputs
        return self.transformer(x, x)  # src and tgt are both the input tensor x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape used in error reproduction examples (1 batch, 10 sequence length, 576 features)
    return torch.rand(1, 10, 576, dtype=torch.float32)

