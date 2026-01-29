import torch
import torch.nn as nn
import torch.distributions as distributions

class RNNEncoder(nn.Module):
    def __init__(self, cond_traj_len, hidden_size):
        super(RNNEncoder, self).__init__()
        self.cond_traj_len = cond_traj_len
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1)
        self.fc_mu = nn.Linear(hidden_size, cond_traj_len)
        self.fc_logvar = nn.Linear(hidden_size, cond_traj_len)

    def forward(self, context, hidden_state=None):
        output, hidden_state = self.gru(context, hidden_state)
        dist_ls = self.hidden_state_to_distribution(output)
        return dist_ls, hidden_state

    def hidden_state_to_distribution(self, hidden_state):
        mean = self.fc_mu(hidden_state)
        log_var = self.fc_logvar(hidden_state)
        return mean, log_var

    def prior(self, batch_size):
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True)
        dist_ls = self.hidden_state_to_distribution(hidden_state)
        return dist_ls, hidden_state

class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask):
        super(RealNVP, self).__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = nn.ModuleList([nets() for _ in range(len(mask))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J = x.new_zeros(x.shape[0])
        z = x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, z_mean, z_logvar, cond_ctx_ls):
        cond_ctx_ls = cond_ctx_ls.flatten(0,1)
        z, logp = self.f(cond_ctx_ls)
        mean = z_mean.flatten(0,1)
        logvar = z_logvar.flatten(0,1)
        prior = distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
            loc=mean,
            cov_factor=torch.zeros(logvar.shape, device=mean.device).unsqueeze(-1),
            cov_diag=torch.exp(logvar),
            validate_args=None
        )
        return prior.log_prob(z) + logp

# torch.rand(B, 200, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        COND_LEN = 50
        HIDDEN_SIZE = 64
        self.encoder = RNNEncoder(cond_traj_len=COND_LEN, hidden_size=HIDDEN_SIZE)
        mask_type_1 = [0]*(COND_LEN//2) + [1]*(COND_LEN - COND_LEN//2)
        mask_type_2 = [1]*(COND_LEN//2) + [0]*(COND_LEN - COND_LEN//2)
        masks = torch.tensor([mask_type_1, mask_type_2]*5, dtype=torch.float32)
        nets = lambda: nn.Sequential(
            nn.Linear(COND_LEN, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, COND_LEN), nn.Tanh()
        )
        nett = lambda: nn.Sequential(
            nn.Linear(COND_LEN, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, COND_LEN)
        )
        self.flow = RealNVP(nets, nett, masks)

    def forward(self, ctx_batch):
        traj_len = 200
        encoder_input = ctx_batch.transpose(0, 1).unsqueeze(-1)
        z_dist_ls, _ = self.encoder(encoder_input)
        z_mean = z_dist_ls[0][:traj_len - 50]  # 50 is COND_LEN
        z_logvar = z_dist_ls[1][:traj_len - 50]

        batch_size = ctx_batch.size(0)
        cond_ctx_ls = torch.zeros((traj_len - 50, batch_size, 50), device=ctx_batch.device)
        for j in range(traj_len - 50):
            cond_ctx_ls[j, :, :] = ctx_batch[:, j:j+50]

        return self.flow.log_prob(z_mean, z_logvar, cond_ctx_ls)

def my_model_function():
    return MyModel()

def GetInput():
    B = 100
    return torch.rand(B, 200, dtype=torch.float32)

