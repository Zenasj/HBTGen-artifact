import torch
import torch.nn as nn

from torch import nn

class Emulator(nn.Module):
    
    __constants__ = ['m', 'M', 'n']
    
    def __init__(self, grid, n_comps=None):
        super().__init__()
        
        self.wl = torch.from_numpy(grid.wl)
        self.grid_points = torch.from_numpy(grid.grid_points)
        self.fluxes = torch.tensor(list(grid.fluxes), dtype=torch.float64)
        self.flux_mean = self.fluxes.mean(0)
        self.flux_sd = self.fluxes.std(0)
        self.pca = PCA(n_comps=n_comps)
        self.components, self.weights = self.pca(self.fluxes)
        self.components.to(torch.float64)
        self.M, self.n = self.grid_points.shape
        self.m = self.weights.shape[-1]
        
        conf_priors = torch.tensor(config.PCA['priors']).t()
        
        self.lam_xi = nn.Parameter(pyro.sample('lam_xi', dist.Gamma(1, 1)))
        l_prior = dist.Gamma(conf_priors[0], conf_priors[1]).expand((self.m, self.n)).to_event(0)
        lengthscale = pyro.sample('lengthscale', l_prior)
        a_prior = dist.Normal(400, 100).expand((self.m,))
        amplitude = pyro.sample('amplitude', a_prior)
        self.kernels = nn.ModuleList()
        self.gprs = nn.ModuleList()
        for i in range(self.m):
            k = gp.kernels.RBF(input_dim=self.n, variance=amplitude[i], lengthscale=lengthscale[i])
            self.kernels.append(k)
            self.gprs.append(gp.models.GPRegression(self.grid_points, self.weights[:, i], k))
        self.PhiPhi = self.components.t().mm(self.components).to(torch.float64)
        self.w_hat = torch.chain_matmul(self.PhiPhi.inverse(), self.components.t(), self.fluxes.t())
    
    def forward(self, params, *args, **kwargs):
        if params.dim() < 2:
            params = params.unsqueeze(0)
        mus = params.new_empty(self.m, params.shape[0])
        covs = params.new_empty(self.m, params.shape[0], params.shape[0])
        for i, gpr in enumerate(self.gprs):
            m, c = gpr(params, *args, **kwargs)
            mus[i] = m
            covs[i] = c
        return mus, covs
    
    def loss(self):
        ws, sig = self(self.grid_points)
        C = (1. / self.lam_xi) * self.PhiPhi
        s, ld = torch.slogdet(C)
        R = ws - self.w_hat
        logl = -0.5 * s * ld - self.lam_xi / 2 * torch.chain_matmul(R.t(), self.PhiPhi, R)
        return -logl.sum()
    
    def load_flux(self, params, get_cov=False):
        mu, cov = self(params, full_cov=get_cov)
        comp = self.components.matmul(mu).t().squeeze()
        if get_cov:
            C = torch.chain_matmul(self.components, cov, self.components.t())
            return comp * self.flux_sd + self.flux_mean, C
        else:
            return comp * self.flux_sd + self.flux_mean

emu = Emulator(grid)
optim = torch.optim.Adam(emu.parameters(), lr=0.01)
num_steps = 100
losses = []
for i in tqdm.trange(num_steps):
    optim.zero_grad()
    loss = emu.loss()
    loss.backward()
    optim.step()
    losses.append(loss.item())