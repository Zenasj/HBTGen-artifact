import torch
from torch.distributions import Normal, TransformedDistribution, AffineTransform, TanhTransform, ComposeTransform

# torch.rand(3, dtype=torch.float32)
class MyModel(torch.nn.Module):
    def forward(self, a):
        base_dist = Normal(a, 1.0)  # Mean from input, fixed std=1
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=2, scale=2)]  # Apply Tanh then Affine
        transform = ComposeTransform(transforms)
        d = TransformedDistribution(base_dist, transform)
        samples = d.rsample(sample_shape=torch.Size([10]))  # Sample shape (10,)
        log_probs = d.log_prob(samples)  # Triggers expand in log_abs_det_jacobian
        return log_probs

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)  # Matches input shape (3,)

