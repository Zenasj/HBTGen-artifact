# torch.rand(B, 4), torch.rand(B, 4, 4), torch.rand(B, m, 4)  # B=batch_size, m=number of measurements
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self._std_weight_position = nn.Parameter(torch.tensor(1e-1), requires_grad=False)
        self._update_mat = nn.Parameter(torch.eye(4), requires_grad=False)  # 4x4 identity matrix

    def project(self, mean, covariance):
        # Compute innovation covariance
        std_template = torch.tensor([[self._std_weight_position.item(),
                                     self._std_weight_position.item(),
                                     1e-1,
                                     self._std_weight_position.item()]],
                                   device=mean.device)
        std = mean[:, [3]] * std_template  # (B,1) * (1,4) → (B,4)
        std[:, 2] = 1e-1  # Force third element to 1e-1
        innovation_cov = torch.diag_embed(std ** 2)

        # Compute updated mean
        update_mat_t = self._update_mat.t()
        new_mean = torch.mm(mean, update_mat_t)

        # Compute updated covariance
        term1 = covariance.permute(0, 2, 1) @ update_mat_t
        term1 = term1.permute(0, 2, 1)
        new_covariance = term1 @ update_mat_t
        new_covariance += innovation_cov

        return new_mean, new_covariance

    def forward(self, inputs, only_position=False):
        mean, covariance, measurements = inputs
        projected_mean, projected_cov = self.project(mean, covariance)
        cholesky_factor = torch.cholesky(projected_cov)
        d = -projected_mean.unsqueeze(1) + measurements  # (B, m, 4)
        z = torch.triangular_solve(d.permute(0, 2, 1), cholesky_factor, upper=False)[0]
        squared_maha = torch.sum(z ** 2, dim=1)  # (B, m)
        return squared_maha

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    m = 2  # Number of measurements per batch
    # Generate mean (B,4)
    mean = torch.rand(B, 4, dtype=torch.float32)
    # Generate positive definite covariance (B,4,4)
    A = torch.rand(B, 4, 4)
    covariance = torch.bmm(A, A.transpose(1, 2))  # A @ A^T
    covariance += 1e-3 * torch.eye(4).unsqueeze(0)  # Ensure PD
    # Generate measurements (B, m, 4)
    measurements = torch.rand(B, m, 4)
    return (mean, covariance, measurements)

