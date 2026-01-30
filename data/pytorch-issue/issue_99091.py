import torch.nn.functional as F
import random

import numpy as np
# from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
from functools import partial

import time
import platform

import torch



def allclose(u, v):
    # we cast to numpy so we can compare pytorch and jax
    return np.allclose(np.array(u), np.array(v), atol=1e-3)


def predict(m, S, F, Q):
    mu_pred = F @ m 
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred


### TORCH

def predict_pt(m, S, F, Q):
    mu_pred = F @ m 
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred

def condition_on_pt(m, P, H, R, y):
    S = R + H @ P @ H.T
    K = torch.linalg.solve(S + 1e-6, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - H @ m)
    return mu_cond, Sigma_cond

def kf_pt(params, emissions, return_covs=False, compile=False):
    F, Q, R = params['F'], params['Q'], params['R']
    def step(carry, t):
        ll, pred_mean, pred_cov = carry
        H = params['Ht'][t]
        y = emissions[t]
        #ll += MVN(H @ pred_mean, H @ pred_cov @ H.T + R).log_prob(y)
        filtered_mean, filtered_cov = condition_on_pt(pred_mean, pred_cov, H, R, y)
        pred_mean, pred_cov = predict_pt(filtered_mean, filtered_cov, F, Q)
        carry = (ll, pred_mean, pred_cov)
        if return_covs:
            return carry, (filtered_mean, filtered_cov)
        else:
            return carry, filtered_mean
    
    if compile:
        if platform.system() == 'Darwin':
            # https://discuss.pytorch.org/t/torch-compile-seems-to-hang/177089
            step = torch.compile(step, backend="aot_eager")
        else:
            import torch._inductor.config
            torch._inductor.config.triton.cudagraphs = True
            torch._inductor.config.pattern_matcher = False
            step = torch.compile(step, dynamic=True)
    num_timesteps = len(emissions)
    D = len(params['mu0'])
    import torch
    filtered_means = torch.zeros((num_timesteps, D))
    if return_covs:
        filtered_covs = torch.zeros((num_timesteps, D, D))
    else:
        filtered_covs = None
    ll = 0
    carry = (ll, params['mu0'], params['Sigma0'])
    for t in range(num_timesteps):
        if return_covs:
            carry, (filtered_means[t], filtered_covs[t]) = step(carry, t)
        else:
            carry, filtered_means[t] = step(carry, t)
    return ll, filtered_means, filtered_covs

## Compare jax and torch

def convert_params_to_pt(params, Y):
    F, Q, R, Ht, mu0, Sigma0 = params['F'], params['Q'], params['R'], params['Ht'], params['mu0'], params['Sigma0']
    Y_pt = torch.tensor(np.array(Y))
    F_pt = torch.tensor(np.array(F))
    Q_pt = torch.tensor(np.array(Q))
    R_pt = torch.tensor(np.array(R))
    Ht_pt = torch.tensor(np.array(Ht))
    mu0_pt = torch.tensor(np.array(mu0))
    Sigma0_pt = torch.tensor(np.array(Sigma0))
    param_dict_pt = {'mu0': mu0_pt, 'Sigma0': Sigma0_pt, 'F': F_pt, 'Q': Q_pt, 'R': R_pt, 'Ht': Ht_pt}
    return param_dict_pt, Y_pt

def make_linreg_data_pt(N, D):
    torch.manual_seed(0)
    X = torch.randn((N, D))
    w = torch.randn((D, 1))
    y = X @ w + 0.1*torch.randn((N, 1))
    return X, y

def make_linreg_data_np(N, D):
    np.random.seed(0)
    X = np.random.randn(N, D)
    w = np.random.randn(D, 1)
    y = X @ w + 0.1*np.random.randn(N, 1)
    return X, y

def make_params_and_data_pt(N, D):
    X_np, Y_np = make_linreg_data_np(N, D)
    N, D = X_np.shape
    X1_np = np.column_stack((np.ones(N), X_np))  # Include column of 1s
    Ht_np = X1_np[:, None, :] # (T,D) -> (T,1,D), yhat = H[t]'z = (b w)' (1 x)
    nfeatures = X1_np.shape[1] # D+1
    Ht_pt = torch.tensor(Ht_np, dtype=torch.double)
    mu0_pt = torch.zeros(nfeatures, dtype=torch.double)
    Sigma0_pt = torch.eye(nfeatures, dtype=torch.double) * 1
    F_pt = torch.eye(nfeatures, dtype=torch.double) # dynamics = I
    Q_pt = torch.zeros((nfeatures, nfeatures), dtype=torch.double)  # No parameter drift.
    R_pt = torch.ones((1, 1), dtype=torch.double) * 0.1
    Y_pt = torch.tensor(Y_np)
    param_dict_pt = {'mu0': mu0_pt, 'Sigma0': Sigma0_pt, 'F': F_pt, 'Q': Q_pt, 'R': R_pt, 'Ht': Ht_pt}
    return param_dict_pt, X_np, Y_pt



def time_torch(N, D, compile=False):
    param_dict_pt, X, Y_pt = make_params_and_data_pt(N, D)
    return_covs = False
    t0 = time.time()
    _ = kf_pt(param_dict_pt, Y_pt, return_covs, compile=compile) 
    return time.time() - t0

def main():
    # torch.set_default_device('cuda')
    N = 100
    D = 500


    compile = False
    runtime_torch = time_torch(N, D, compile=compile)
    print('torch, time={:.3f} compile {} N {} D {}'.format(runtime_torch, compile, N, D))

    # Warmup
    compile = True
    runtime_torch = time_torch(N, D, compile=compile)
    print('torch, time={:.3f} compile {} N {} D {}'.format(runtime_torch, compile, N, D))

    # Actual compilation time
    runtime_torch = time_torch(N, D, compile=compile)
    print('torch, time={:.3f} compile {} N {} D {}'.format(runtime_torch, compile, N, D))

if __name__ == '__main__':
    main()