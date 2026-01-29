# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, n_in_channel: int, n_out_channel: int, kernel, stride, padding, n_layer: int = 1, on_batch_norm: bool = True, activation_func: str = "relu"):
        super(MyModel, self).__init__()
        self.n_in_channel = n_in_channel
        self.n_out_channel = n_out_channel
        self.n_layer = n_layer
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.on_batch_norm = on_batch_norm

        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(n_out_channel) for _ in range(n_layer)])
        self.conv2ds = nn.ModuleList([nn.Conv2d(self.n_in_channel, self.n_out_channel, self.kernel, self.stride, self.padding)])
        for i in range(n_layer - 1):
            self.conv2ds.append(nn.Conv2d(self.n_out_channel, self.n_out_channel, self.kernel, self.stride, self.padding))

        self.activation_funcs = nn.ModuleList([get_activation_func(activation_func) for _ in range(n_layer)])

        self.seq = nn.Sequential()
        for i in range(n_layer):
            self.seq.append(self.conv2ds[i])
            if self.on_batch_norm:
                self.seq.append(self.batch_norms[i])
            self.seq.append(self.activation_funcs[i])

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(n_in_channel=3, n_out_channel=64, kernel=3, stride=1, padding=1, n_layer=2, on_batch_norm=True, activation_func="relu")

def get_activation_func(activation_func: str):
    if activation_func == "relu":
        return nn.ReLU()
    # Add more activation functions as needed
    else:
        raise ValueError(f"Unsupported activation function: {activation_func}")

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 224, 224  # Example batch size, channels, height, width
    return torch.rand(B, C, H, W, dtype=torch.float32)

