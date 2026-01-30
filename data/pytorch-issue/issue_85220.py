import torch.nn as nn
import random

# [x] raise dimension to 50
# [x] use a real dataset
# [ ] add RNN as the condition
import shutil
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import datetime
import platform
import os
import time
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler


# load csv file data.csv
sales_mat = np.random.normal(size=(500, 1000))
sales_mat_norm = ((sales_mat.T - np.mean(sales_mat,axis=1))/np.std(sales_mat,axis=1)).T


class ContextIterator():
    def __init__(self, context_mat, traj_len):
        self.context_mat = torch.tensor(context_mat.astype(np.float32)).to(device)[:,:traj_len]
        self.n_traj = context_mat.shape[0]
        self.traj_len = self.context_mat.shape[1]
        
    def sample(self, batch_size):
        if batch_size >= self.n_traj:
            return self.context_mat
        else:
            idx = np.random.randint(0, self.n_traj, batch_size)
            return self.context_mat[idx,:]
            

def sample_data(n_samples, name, dim):
    if name == 'moon':
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)[0].astype(np.float32)
        return noisy_moons
    elif name == 'sales':
        if n_samples >= 1000:
            return sales_mat_norm[:,:dim].astype(np.float32)
        else:
            idx = np.random.randint(1000,size=n_samples)
            return sales_mat_norm[idx, :dim].astype(np.float32)
            
DATASET = 'sales'
BATCH_SIZE = 100
# COND_LEN = 50
LR = 1e-6
ITERS = 100000
GPU = True

COND_LEN = 50
VIS_INTERVAL = 50
HIDDEN_SIZE = 64 # RNN hidden size

if platform.system() == 'Darwin':
    result_dir = '/Users/shuffleofficial/Offline_Documents/FutureRL/tmp_results'
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # import torch after setting PYTORCH_ENABLE_MPS_FALLBACK
    import torch
    from torch import nn
    from torch import distributions

    if GPU:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
else:
    result_dir = '/home/yufeng/FutureRL_tmp'
time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_dir = os.path.join(result_dir, 'NF_'+time_str)

# copy all the files in cwd to result_dir, except for the 'data.csv' file
shutil.copytree('.', result_dir, ignore=shutil.ignore_patterns('data.csv'))


class RNNEncoder(nn.Module):
    def __init__(self, cond_traj_len, hidden_size):
        super(RNNEncoder, self).__init__()
        self.cond_traj_len = cond_traj_len
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size=1, # context dimension for one step
                          hidden_size=hidden_size,
                          num_layers=1).to(device)
        
        # output layer
        self.fc_mu = nn.Linear(hidden_size, cond_traj_len).to(device)
        self.fc_logvar = nn.Linear(hidden_size, cond_traj_len).to(device)
        
    def forward(self, context, hidden_state=None):
        # context: [seq_len, batch_size, ctx_dim]
        output, hidden_state = self.gru(input=context, hx=hidden_state) # output == hidden_state
        dist_ls = self.hidden_state_to_distribution(output)
        return dist_ls, hidden_state
    
    def hidden_state_to_distribution(self, hidden_state):
        # hidden_state: [seq_len, batch_size, hidden_size]
        mean = self.fc_mu(hidden_state)
        log_var = self.fc_logvar(hidden_state)
        return mean, log_var
        
    def prior(self, batch_size):
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)
        dist_ls = self.hidden_state_to_distribution(hidden_state)
        return dist_ls, hidden_state

    
encoder = RNNEncoder(cond_traj_len=COND_LEN, hidden_size=HIDDEN_SIZE)

ctx_iter = ContextIterator(context_mat=sales_mat_norm, traj_len=200)


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask):
        super(RealNVP, self).__init__()
        # self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self, z_mean, z_logvar, cond_ctx_ls):
        # cond_ctx_ls: [traj_len - cond_len, batch_size, cond_len]
        cond_ctx_ls = cond_ctx_ls.flatten(0,1) # [(traj_len - cond_len) * batch_size, cond_len]
        z, logp = self.f(cond_ctx_ls)
        mean = z_mean.flatten(0,1)
        logvar = z_logvar.flatten(0,1)
        prior = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(loc=mean,
                                                                                          cov_factor=torch.zeros(logvar.shape, device=device).unsqueeze(-1),
                                                                                          cov_diag=torch.exp(logvar), validate_args=None)
        return prior.log_prob(z) + logp
        
    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        # logp = self.prior.log_prob(z)
        x = self.g(z)
        return x
    
def get_CI(data_mat):
    mean = np.mean(data_mat.detach().cpu().numpy(), axis=0)
    std = np.std(data_mat.detach().cpu().numpy(), axis=0)
    ub = mean + 1.96 * std
    lb = mean - 1.96 * std
    return lb, ub
    

def visualize(step, img_dir_name):
    ctx_traj = ctx_iter.sample(1)
    step_idx = np.random.randint(ctx_iter.traj_len - COND_LEN)
    # [seq_len, batch_size, ctx_dim]
    z_dist_ls, hidden_state = encoder.forward(ctx_traj.transpose(0,1).unsqueeze(-1), None)
    # sample from 
    mean = z_dist_ls[0][step_idx, 0, :]
    logvar = z_dist_ls[1][step_idx, 0, :]
    VIS_BATCH_SIZE = 1000
    prior = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(loc=mean,
                                                                                        cov_factor=torch.zeros(logvar.shape, device=device).unsqueeze(-1),
                                                                                        cov_diag=torch.exp(logvar), validate_args=None)
    z = prior.sample(sample_shape=torch.Size([VIS_BATCH_SIZE]))
    samples = flow.g(z)
    lb, ub = get_CI(samples)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(np.arange(step_idx+COND_LEN) - step_idx, ctx_traj[0,:step_idx+COND_LEN].cpu())
    plt.plot(samples.detach().cpu().T, c='r', alpha=0.01)
    plt.plot(lb, c='k')
    plt.plot(ub, c='k')
    plt.subplot(122)
    plt.plot(mean.detach().cpu().numpy(), label='prior mean')
    plt.plot(np.exp(logvar.detach().cpu().numpy()), label='prior var')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir_name, 'step_{}.png'.format(step)))
    plt.close('all')
    

def visualize_old(step, img_dir_name):
    VIS_BATCH_SIZE = 1000
    noisy_moons = sample_data(VIS_BATCH_SIZE, DATASET, COND_LEN)
    z = flow.f(torch.from_numpy(noisy_moons).to(device))[0].detach().cpu().numpy()
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z = f(X)$')

    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), VIS_BATCH_SIZE)
    plt.subplot(222)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z \sim p(z)$')

    plt.subplot(223)
    real = sample_data(VIS_BATCH_SIZE, DATASET, COND_LEN)
    plt.scatter(real[:, 0], real[:, 1], c='r')
    plt.title(r'$X \sim p(X)$')

    plt.subplot(224)
    x = flow.sample(VIS_BATCH_SIZE).detach().cpu().numpy()
    plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
    plt.title(r'$X = g(z)$')
    plt.tight_layout()
    plt.savefig(img_dir_name + '/scatter_%s.png' % step)
    plt.close('all')
    
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.plot(np.mean(x,axis=0)[0],label='fake')
    plt.plot(np.mean(real,axis=0),label='real')
    plt.legend()
    plt.title('Marginal mean')
    
    plt.subplot(222)
    plt.plot(np.var(x,axis=0)[0],label='fake')
    plt.plot(np.var(real,axis=0),label='real')
    plt.legend()
    plt.title('marginal variance')
    
    plt.subplot(223)
    fake_corr = np.corrcoef(x[:,0,:].T)
    real_corr = np.corrcoef(real.T)
    vmin = min(np.min(fake_corr), np.min(real_corr))
    vmax = max(np.max(fake_corr), np.max(real_corr))
    plt.pcolormesh(fake_corr, vmin=vmin, vmax=vmax)
    plt.title('Fake Coefficient')
    
    plt.subplot(224)
    plt.pcolormesh(real_corr, vmin=vmin, vmax=vmax)
    plt.title('Real corrcoef')
    plt.savefig(img_dir_name + '/marginal_%s.png' % step)
    plt.close('all')

nets = lambda: nn.Sequential(nn.Linear(COND_LEN, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, COND_LEN), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(COND_LEN, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, COND_LEN))

mask_type_1 = [0] * (COND_LEN // 2) + [1] * (COND_LEN - (COND_LEN // 2))
mask_type_2 = [1] * (COND_LEN // 2) + [0] * (COND_LEN - (COND_LEN // 2))

masks = torch.from_numpy(np.array([mask_type_1, mask_type_2] * 5).astype(np.float32))
# prior = distributions.MultivariateNormal(torch.zeros(COND_LEN, device=device), torch.eye(COND_LEN, device=device))
flow = RealNVP(nets, nett, masks).to(device)

log_prob_rec = []
time_rec = []

start_time = time.time()
optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True] + \
    [p for p in encoder.parameters() if p.requires_grad==True], lr=LR)


for t in progressbar.progressbar(range(ITERS+1), redirect_stdout=True):  
    ctx_batch = ctx_iter.sample(BATCH_SIZE)
    # z_dist_ls: torch.Size([traj_len, batch_size, cond_dim])
    z_dist_ls, hidden_state = encoder.forward(ctx_batch.transpose(0,1).unsqueeze(-1), None)
    # cond_ctx_ls: [traj_len - cond_dim, batch_size, cond_dim]
    z_mean = z_dist_ls[0][:ctx_iter.traj_len - COND_LEN]
    z_logvar = z_dist_ls[1][:ctx_iter.traj_len - COND_LEN]
    cond_ctx_ls = torch.zeros((ctx_iter.traj_len - COND_LEN, BATCH_SIZE, COND_LEN), device=device)
    for j in range(ctx_iter.traj_len - COND_LEN):
        cond_ctx_ls[j,:,:] = ctx_batch[:,j:j+COND_LEN]
    
    # noisy_moons = sample_data(BATCH_SIZE, DATASET, COND_LEN)
    loss = -flow.log_prob(z_mean, z_logvar, cond_ctx_ls).mean()
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    log_prob_rec.append(-loss.detach().cpu())
    time_rec.append(time.time() - start_time)
    
    if t % VIS_INTERVAL == 0:
        print('iter %s:' % t, 'log prob = %.3f' % -loss)
        visualize(t, result_dir)
        plt.figure(figsize=(10,5))
        plt.subplot(211)
        plt.plot(log_prob_rec)
        plt.xlabel('Iters')
        plt.subplot(212)
        plt.plot(time_rec, log_prob_rec)
        plt.xlabel('Wall clock time (s)')
        plt.savefig(os.path.join(result_dir, 'log_prob_rec'))
        plt.close()