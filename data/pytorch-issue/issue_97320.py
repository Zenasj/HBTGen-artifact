# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (20, 5, 10, dtype=torch.int32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_norm = nn.LayerNorm(10)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            ret = self.layer_norm(x.to(torch.float32)).to(x.dtype)
        return ret

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(20, 5, 10).int().cuda()

