# torch.rand(B, 32, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)

    @staticmethod
    def quantize(x):
        # Placeholder quantization (e.g., 8-bit compression)
        # Actual implementation would involve scaling and rounding
        return x  # Identity for demonstration

    @staticmethod
    def dequantize(x):
        # Placeholder dequantization
        return x  # Identity for demonstration

    @staticmethod
    def pack_hook(x):
        if isinstance(x, torch.nn.Parameter):
            return (x, "not_packed")
        else:
            return (MyModel.quantize(x), "packed")

    @staticmethod
    def unpack_hook(packed):
        x, status = packed
        if status == "not_packed":
            return x
        else:
            return MyModel.dequantize(x)

    def forward(self, x):
        with torch.autograd.graph.saved_tensors_hooks(
            MyModel.pack_hook, MyModel.unpack_hook
        ):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x

def my_model_function():
    # Returns MyModel instance with quantization hooks
    return MyModel()

def GetInput():
    # Returns random input tensor matching (B, 32, H, W) shape
    return torch.rand(2, 32, 32, 32, dtype=torch.float32)

