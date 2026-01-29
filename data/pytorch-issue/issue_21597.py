# torch.rand(1, 6, 5, 5, dtype=torch.float)  # Add a comment line at the top with the inferred input shape

import torch
import numpy as np

class MyModel(torch.nn.Module):
    def __init__(self, input_channels=6, output_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=2):
        super(MyModel, self).__init__()
        self.conv_op = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups
        )
        W_value_min = 0
        W_value_max = 5
        W_init = torch.from_numpy(
            np.random.randint(
                W_value_min,
                W_value_max,
                (output_channels, input_channels // groups, kernel_size[0], kernel_size[1]),
            )
        ).to(dtype=torch.float)
        b_init = torch.from_numpy(np.zeros((output_channels,))).to(dtype=torch.float)
        
        self.conv_op.weight = torch.nn.Parameter(W_init, requires_grad=False)
        self.conv_op.bias = torch.nn.Parameter(b_init, requires_grad=False)

    def forward(self, x):
        return self.conv_op(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 1
    input_channels = 6
    height = 5
    width = 5
    X_value_min = 0
    X_value_max = 5
    X_init = torch.from_numpy(
        np.random.randint(
            X_value_min, X_value_max, (batch_size, input_channels, height, width)
        )
    ).to(dtype=torch.float)
    return X_init

