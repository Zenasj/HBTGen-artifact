# torch.rand(1, 2, 1, 3, 1, 2, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.distributions as dist

class MyModel(nn.Module):
    def forward(self, logits):
        m = dist.Categorical(logits=logits)
        return m.sample(torch.Size([4]))  # Fixed sample_shape as in the test case

def my_model_function():
    return MyModel()

def GetInput():
    batch_shape = (1, 2, 1, 3, 1)
    cardinality = 2
    shape = batch_shape + (cardinality,)
    return torch.rand(shape, dtype=torch.float32)

