# torch.rand(100, 1, dtype=torch.double)  # N is the number of time steps
import torch
from torch import nn
import numpy as np
from typing import NamedTuple, Tuple, Any

def allclose(u, v):
    return np.allclose(np.array(u), np.array(v), atol=1e-3)

def make_linreg_data_np(N, D):
    np.random.seed(0)
    X = np.random.randn(N, D)
    w = np.random.randn(D, 1)
    y = X @ w + 0.1 * np.random.randn(N, 1)
    return X, y

def make_linreg_data_pt(N, D):
    torch.manual_seed(0)
    X = torch.randn((N, D))
    w = torch.randn((D, 1))
    y = X @ w + 0.1 * torch.randn((N, 1))
    return X, y

def make_params_and_data_pt(N, D):
    X_np, Y_np = make_linreg_data_np(N, D)
    N, D = X_np.shape
    X1_np = np.column_stack((np.ones(N), X_np))
    Ht_np = X1_np[:, None, :]
    nfeatures = X1_np.shape[1]
    Ht_pt = torch.tensor(Ht_np, dtype=torch.double)
    mu0_pt = torch.zeros(nfeatures, dtype=torch.double)
    Sigma0_pt = torch.eye(nfeatures, dtype=torch.double) * 1
    F_pt = torch.eye(nfeatures, dtype=torch.double)
    Q_pt = torch.zeros((nfeatures, nfeatures), dtype=torch.double)
    R_pt = torch.ones((1, 1), dtype=torch.double) * 0.1
    Y_pt = torch.tensor(Y_np, dtype=torch.double)
    param_dict_pt = {
        'mu0': mu0_pt,
        'Sigma0': Sigma0_pt,
        'F': F_pt,
        'Q': Q_pt,
        'R': R_pt,
        'Ht': Ht_pt
    }
    return param_dict_pt, X_np, Y_pt

class MyModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.register_buffer('F', params['F'])
        self.register_buffer('Q', params['Q'])
        self.register_buffer('R', params['R'])
        self.register_buffer('Ht', params['Ht'])
        self.register_buffer('mu0', params['mu0'].unsqueeze(1))  # (D, 1)
        self.register_buffer('Sigma0', params['Sigma0'])  # (D, D)

    def forward(self, emissions):
        N, _ = emissions.shape
        D = self.mu0.shape[0]
        filtered_means = []
        pred_mean = self.mu0  # (D, 1)
        pred_cov = self.Sigma0  # (D, D)

        for t in range(N):
            H = self.Ht[t]  # (1, D)
            y = emissions[t]  # (1, )

            # Compute S = R + H @ pred_cov @ H.T + 1e-6*I
            S = self.R + H @ pred_cov @ H.T
            S += 1e-6 * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)

            # Compute K = (H @ pred_cov) / S[0,0]
            numerator = H @ pred_cov
            denominator = S[0, 0]
            x = numerator / denominator
            K = x.T  # (D, 1)

            # Compute innovation
            innovation = y - H @ pred_mean  # (1,1)

            # Update mean and covariance
            mu_cond = pred_mean + K @ innovation
            Sigma_cond = pred_cov - K @ S @ K.T

            # Predict step
            mu_pred = self.F @ mu_cond
            Sigma_pred = self.F @ Sigma_cond @ self.F.T + self.Q

            # Save results
            filtered_means.append(mu_cond.squeeze(1))

            # Update for next iteration
            pred_mean = mu_pred
            pred_cov = Sigma_pred

        return torch.stack(filtered_means, dim=0)

def my_model_function():
    param_dict, _, _ = make_params_and_data_pt(100, 500)
    return MyModel(param_dict)

def GetInput():
    _, _, Y_pt = make_params_and_data_pt(100, 500)
    return Y_pt

