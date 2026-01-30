import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import optim, nn
from torch.nn import functional; nn.f = functional
from torch.func import vmap, jacrev


class VariationalAutoencoder(nn.Module):
    def __init__(self, w=200, D=20):
        super().__init__()
        self.D = D    # latent size
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
        μ_lσ2 = h.view(-1, 2, self.D)
        μ      = μ_lσ2[:, 0, :]
        log_σ2 = μ_lσ2[:, 1, :]
        ζ = self.reparameterise(μ, log_σ2)
        ỹ = self.decoder(ζ)
        if self.training:
            return ỹ, μ, log_σ2
        return ỹ


def vmap_test(net):
    n = lambda b: nn.f.normalize(b, dim=-1)
    u = lambda ȳ: n(net(ȳ) - ȳ)
    with torch.no_grad():
        u_Ȳ = u(Ȳ)
        J_u_Ȳ = vmap(jacrev(u))(Ȳ)
    print(u_Ȳ.size(), J_u_Ȳ.size())


Ȳ = torch.randn(961, 2)

vae = VariationalAutoencoder()
vae.eval()

vmap_test(vae)

def u(y):
    """
    y: D, 2
    returns: D, 2
    """
    net = VariationalAutoencoder()
    net.eval()
    with torch.no_grad():
        return F.normalize(net(y) - y, dim=-1)

def ju(y):
    """
    y: D, 2
    returns: D, 2, D, 2
    """
    return jacrev(u)(y)

D = 961
B = 10

y = torch.randn(D, 2)
ys = torch.randn(B, D, 2)

uy = u(y)
uys = vmap(u, randomness='same')(ys)

juy = ju(y)
juys = vmap(ju, randomness='same')(ys)

for name, var in dict(y=y, uy=uy, juy=juy, ys=ys, uys=uys, juys=juys).items():
    print(f'{name}.shape = {var.shape}')

assert tuple(juys.shape) == (B, *juy.shape)
assert tuple(uys.shape) == (B, *uy.shape)

μ_lσ2 = h.unflatten(-1, (2, self.D))
μ      = μ_lσ2.select(-2, 0)
log_σ2 = μ_lσ2.select(-2, 1)

μ_lσ2  = h.view(-1, 2, self.D)
μ      = μ_lσ2[:, 0, :]
log_σ2 = μ_lσ2[:, 1, :]

μ_lσ2  = h.unflatten(-1, (2, self.D))
μ      = μ_lσ2.select(-2, 0)
log_σ2 = μ_lσ2.select(-2, 1)