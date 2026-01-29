# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, w=200, D=20):
        super().__init__()
        self.D = D
        NL = nn.ReLU  # non-linearity
        self.encoder = nn.Sequential(
            nn.Linear(2, w), NL(),
            nn.Linear(w, w), NL(),
            nn.Linear(w, 2 * D),
        )
        self.decoder = nn.Sequential(
            nn.Linear(D, w), NL(),
            nn.Linear(w, w), NL(),
            nn.Linear(w, 2)
        )

    def reparameterise(self, μ, log_σ2):
        if self.training:
            σ = log_σ2.mul(0.5).exp_()
            z = torch.randn(self.D, device=σ.device)
            return z.mul(σ).add_(μ)
        return μ

    def forward(self, y):
        h = self.encoder(y)
        μ_lσ2 = h.unflatten(-1, (2, self.D))  # Fix from issue discussion
        μ = μ_lσ2.select(-2, 0)
        log_σ2 = μ_lσ2.select(-2, 1)
        ζ = self.reparameterise(μ, log_σ2)
        ỹ = self.decoder(ζ)
        if self.training:
            return ỹ, μ, log_σ2
        return ỹ

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(961, 2, dtype=torch.float32)

