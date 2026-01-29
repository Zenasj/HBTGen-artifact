# torch.rand(B, n, dtype=torch.float64)
import torch
import torch.nn as nn

class DummyGPR(nn.Module):
    def __init__(self):
        super().__init__()
        # Add minimal parameter to satisfy module requirements
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x):
        batch_size = x.size(0)
        return torch.zeros(batch_size, dtype=torch.float64), torch.eye(batch_size, dtype=torch.float64)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inferred parameters based on original code's structure
        self.m = 3  # Number of PCA components
        self.n = 2  # Dimensionality of input parameters
        self.M = 10  # Number of grid points
        self.components = nn.Parameter(torch.randn(5, self.m, dtype=torch.float64), requires_grad=False)  # Example flux dimension 5
        self.PhiPhi = self.components.t().mm(self.components)
        self.w_hat = torch.randn(self.m, self.M, dtype=torch.float64)  # Precomputed weights
        self.lam_xi = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))  # Regularization parameter
        self.grid_points = torch.randn(self.M, self.n, dtype=torch.float64)  # Dummy grid points
        self.gprs = nn.ModuleList([DummyGPR() for _ in range(self.m)])  # Mocked GP models

    def forward(self, params, *args, **kwargs):
        if params.dim() < 2:
            params = params.unsqueeze(0)
        batch_size = params.size(0)
        mus = params.new_zeros((self.m, batch_size), dtype=torch.float64)
        covs = params.new_zeros((self.m, batch_size, batch_size), dtype=torch.float64)
        for i, gpr in enumerate(self.gprs):
            m, c = gpr(params)
            mus[i] = m
            covs[i] = c
        return mus, covs

    def loss(self):
        ws, _ = self(self.grid_points)
        C = (1.0 / self.lam_xi) * self.PhiPhi
        s, ld = torch.slogdet(C)
        R = ws - self.w_hat
        logl = -0.5 * s * ld - self.lam_xi / 2 * torch.chain_matmul(R.t(), self.PhiPhi, R)
        return -logl.sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching input shape (B, n)
    return torch.rand(1, MyModel().n, dtype=torch.float64)

