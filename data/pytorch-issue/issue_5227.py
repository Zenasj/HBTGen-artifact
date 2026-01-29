# torch.rand(1, 4, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.distribution = torch.distributions.Categorical(torch.Tensor([0.25, 0.25, 0.25, 0.25]))

    def forward(self, x):
        # Sample from the categorical distribution
        sample = self.distribution.sample(sample_shape=x.shape[:-1])
        return sample

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input shape is (batch_size, 4) where 4 is the number of categories
    batch_size = 1
    return torch.rand(batch_size, 4, dtype=torch.float32)

