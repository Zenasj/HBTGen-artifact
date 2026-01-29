# torch.rand(100, 1, dtype=torch.double)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, D=500, N=100):
        super().__init__()
        self.D = D  # Original feature dimension (X has D features)
        self.N = N  # Number of time steps
        # Initialize parameters as per make_params_and_data_pt
        self.register_buffer('Ht', torch.randn(N, 1, D+1, dtype=torch.double))
        self.F = nn.Parameter(torch.eye(D+1, dtype=torch.double))
        self.Q = nn.Parameter(torch.zeros(D+1, D+1, dtype=torch.double))
        self.R = nn.Parameter(0.1 * torch.eye(1, dtype=torch.double))
        self.mu0 = nn.Parameter(torch.zeros(D+1, dtype=torch.double))
        self.Sigma0 = nn.Parameter(torch.eye(D+1, dtype=torch.double))

    def forward(self, emissions):
        F = self.F
        Q = self.Q
        R = self.R
        Ht = self.Ht
        mu0 = self.mu0
        Sigma0 = self.Sigma0

        num_timesteps = emissions.size(0)
        D_plus_1 = self.D + 1
        filtered_means = torch.zeros((num_timesteps, D_plus_1), dtype=torch.double)
        ll = 0.0
        carry = (ll, mu0, Sigma0)

        for t in range(num_timesteps):
            H = Ht[t]  # (1, D_plus_1)
            y = emissions[t].unsqueeze(0)  # (1,)

            # Predict step
            pred_mean, pred_cov = self.predict_step(carry[1], carry[2], F, Q)

            # Update step
            filtered_mean, filtered_cov = self.condition_step(pred_mean, pred_cov, H, R, y)

            # Update carry
            carry = (ll, filtered_mean, filtered_cov)
            filtered_means[t] = filtered_mean

        return filtered_means

    def predict_step(self, m, S, F, Q):
        mu_pred = F @ m.unsqueeze(1).double()  # (D+1, 1)
        mu_pred = mu_pred.squeeze(1)  # (D+1,)
        Sigma_pred = F @ S @ F.T + Q
        return mu_pred, Sigma_pred

    def condition_step(self, m, S, H, R, y):
        H = H.squeeze(0).unsqueeze(0)  # Ensure (1, D+1) shape
        S_term = R + H @ S @ H.T
        A = S_term + 1e-6 * torch.eye(1, dtype=S_term.dtype, device=S_term.device)
        B = H @ S  # (1, D+1)
        K_T = torch.linalg.solve(A, B)
        K = K_T.T  # (D+1, 1)
        Sigma_cond = S - K @ S_term @ K.T
        innovation = y - H @ m.unsqueeze(1)
        mu_cond = m + (K @ innovation).squeeze(1)
        return mu_cond, Sigma_cond

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, 1, dtype=torch.double)

