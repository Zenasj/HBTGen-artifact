from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal, OneHotCategorical
import torch
import torch.nn as nn
class NormalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_components, full_cov=True):
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.full_cov = full_cov
        self.tril_indices = torch.tril_indices(row=out_dim, col=out_dim, offset=0)
        self.elu = nn.ELU()
        self.mean_net = nn.Sequential(
                nn.Linear(in_dim, out_dim * n_components),
            )
        if full_cov:
            # Cholesky decomposition of the covariance matrix
            self.tril_net = nn.Sequential(
                nn.Linear(in_dim, int(out_dim * (out_dim + 1) / 2 * n_components)),
            )
        else:
            self.tril_net = nn.Sequential(
                nn.Linear(in_dim, out_dim * n_components),
            )
    def forward(self, x):
        mean = self.mean_net(x).reshape(-1, self.n_components, self.out_dim)
        if self.full_cov:
            tril_values = self.tril_net(x).reshape(mean.shape[0], self.n_components, -1)
            tril = torch.zeros(mean.shape[0], mean.shape[1], mean.shape[2], mean.shape[2]).to(x.device)
            tril[:, :, self.tril_indices[0], self.tril_indices[1]] = tril_values
            tril = tril - torch.diag_embed(torch.diagonal(tril, dim1=-2, dim2=-1)) + torch.diag_embed(self.elu(torch.diagonal(tril, dim1=-2, dim2=-1)) + 1 + 1e-8)
        else:
            tril = self.tril_net(x).reshape(mean.shape[0], self.n_components, -1)
            tril = torch.diag_embed(self.elu(tril) + 1 + 1e-8)
        return MultivariateNormal(mean, scale_tril=tril)

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)

class VGMM(nn.Module):
    def __init__(self,
                         u_dim,
                         y_dim,
                         h_dim,
                         n_mixtures):
        super(VGMM, self).__init__()

        self.y_dim = y_dim
        self.u_dim = u_dim
        self.h_dim = h_dim
        self.n_mixtures = n_mixtures
        ## (decoder) Normal distribution
        self.decoder_normal = NormalNetwork(self.h_dim, self.y_dim , self.n_mixtures)

        ##
        #(decoder) Categorical distribution
        self.dec_pi = CategoricalNetwork(self.h_dim, self.n_mixtures)
    def forward(self, u, y):
                dec_normal_t = self.decoder_normal(decoder_out)
                dec_pi_t     = self.dec_pi(decoder_out)
                loss_pred = self.loglikelihood_gmm(y[:, :, t], dec_normal_t, dec_pi_t)

    def loglikelihood_gmm(self, y, normal, pi):
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        assert not torch.isnan(loss).any()
        return loss.mean()