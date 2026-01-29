import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

# torch.rand(B, 2, dtype=torch.float32, device='cuda')
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.mu = nn.Parameter(torch.ones(2, device='cuda'), requires_grad=False)
        self.scale_tril = nn.Parameter(torch.eye(2, device='cuda'), requires_grad=False)
        self.cov_matrix = self.scale_tril @ self.scale_tril.t()  # Precompute covariance matrix
        # Initialize both distributions as submodules
        self.dist_scale_tril = MultivariateNormal(loc=self.mu, scale_tril=self.scale_tril)
        self.dist_covariance = MultivariateNormal(loc=self.mu, covariance_matrix=self.cov_matrix)

    def forward(self, x):
        # Compute log_prob for both distributions
        log_prob_scale = self.dist_scale_tril.log_prob(x)
        log_prob_cov = self.dist_covariance.log_prob(x)
        # Return both outputs for comparison (issue's core comparison)
        return log_prob_scale, log_prob_cov

def my_model_function():
    # Returns the fused model comparing both initialization methods
    return MyModel()

def GetInput():
    # Matches input expected by MyModel (batch_size x 2 features on CUDA)
    return torch.rand(1, 2, dtype=torch.float32, device='cuda')

