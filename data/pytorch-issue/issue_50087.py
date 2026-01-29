# torch.rand(B, 3, 224, 224, dtype=torch.float)  # Input shape inferred as standard image dimensions
import math
import torch
import torch.nn as nn

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super().__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        # Register random matrices as buffers to ensure they move with the module
        for i in range(self.input_num):
            matrix = torch.rand(input_dim_list[i], output_dim)
            self.register_buffer(f'random_matrix_{i}', matrix)

    def forward(self, input_list):
        return_list = []
        for i in range(self.input_num):
            matrix = getattr(self, f'random_matrix_{i}')
            return_list.append(torch.mm(input_list[i], matrix))
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulated branches to produce inputs for RandomLayer (4096 and 384 dimensions)
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 4096)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 384)
        )
        # Use corrected RandomLayer with registered buffers
        self.random_layer = RandomLayer([4096, 384], 2048)

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        return self.random_layer([feat1, feat2])

def my_model_function():
    return MyModel()

def GetInput():
    # Random input tensor matching the expected shape (B, C, H, W)
    return torch.rand(2, 3, 224, 224, dtype=torch.float)

