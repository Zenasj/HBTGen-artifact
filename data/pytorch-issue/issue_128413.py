# torch.rand(B, seq_len, d_model, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True)
        self.enc_layer.eval()  # Ensure fast path conditions are met
        self.cache = []
        self.enc_layer.self_attn.register_forward_hook(self.hook)
    
    def hook(self, module, inputs, output):
        self.cache.append(output[0].detach())  # Capture self-attention output
    
    def forward(self, x):
        self.cache.clear()  # Reset cache for each forward pass
        with torch.inference_mode():
            _ = self.enc_layer(x)  # Run the layer in inference mode
        # Return boolean indicating if hook was called (cache is non-empty)
        return torch.tensor(len(self.cache) > 0, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, 6, 32, dtype=torch.float32)

