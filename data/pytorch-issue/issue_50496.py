import torch
import torch.distributions as dist
from torch.distributions.constraint_registry import biject_to

# torch.rand(3, 2, dtype=torch.float32)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_dist = dist.Exponential(torch.ones(2))
        self.broken_dist = dist.Independent(self.base_dist, 1)

    def forward(self, x):
        tform = biject_to(self.broken_dist.support)
        y = tform(x)
        jac = tform.log_abs_det_jacobian(x, y)
        correct_shape = (x.shape[0],)  # Expected [3] after fix
        is_correct = (jac.shape == correct_shape)
        return torch.tensor(is_correct, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 2)

