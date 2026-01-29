# torch.rand(B, 11, dtype=torch.float32)  # Input shape: [batch_size, 11]
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_1_canonical = nn.Parameter(torch.rand(2, 2))  # 2x2 matrix
        self.b_1_canonical = nn.Parameter(torch.rand(2))     # 2-element vector
        self.A_2_canonical = nn.Parameter(torch.rand(2, 2))  # 2x2 matrix
        self.b_2_canonical = nn.Parameter(torch.rand(2))     # 2-element vector
        self.num_1_points = 2
        self.num_2_points = 2  # Assumed based on context

    def forward(self, x):
        B = x.size(0)
        p1_tilde = x[:, 0:2]
        p2_tilde = x[:, 2:4]
        lambda_1_tilde = x[:, 4:6]
        lambda_2_tilde = x[:, 6:8]
        q_pos1 = x[:, 8:10]
        q_theta = x[:, 10]

        # Rotation matrix construction
        cos_theta = torch.cos(q_theta)
        sin_theta = torch.sin(q_theta)
        rot_mat = torch.zeros(B, 2, 2, dtype=q_theta.dtype, device=q_theta.device)
        rot_mat[:, 0, 0] = cos_theta
        rot_mat[:, 0, 1] = -sin_theta
        rot_mat[:, 1, 0] = sin_theta
        rot_mat[:, 1, 1] = cos_theta
        rot_mat_inv = torch.linalg.inv(rot_mat)

        # Compute box1_A and box1_b
        A1 = self.A_1_canonical.unsqueeze(0).expand(B, 2, 2)
        box1_A = torch.bmm(A1, rot_mat_inv)
        box1_term = torch.bmm(box1_A, q_pos1.unsqueeze(-1)).squeeze(-1)
        box1_b = self.b_1_canonical + box1_term

        # Initialize K tensor
        D = 2
        K_size = 2 * D + self.num_1_points + self.num_2_points
        K = torch.zeros(B, K_size, dtype=x.dtype, device=x.device)

        # Compute K components
        # Lagrangian gradient for p1
        term_p1 = torch.bmm(box1_A.transpose(1, 2), lambda_1_tilde.unsqueeze(-1)).squeeze(-1)
        K[:, :D] = 2 * (p1_tilde - p2_tilde) + term_p1

        # Lagrangian gradient for p2
        A2_T = self.A_2_canonical.t().unsqueeze(0).expand(B, 2, 2)
        term_p2 = torch.bmm(A2_T, lambda_2_tilde.unsqueeze(-1)).squeeze(-1)
        K[:, D:2*D] = -2 * (p1_tilde - p2_tilde) + term_p2

        # Lagrangian gradient for lambda1
        box1_A_p1 = torch.bmm(box1_A, p1_tilde.unsqueeze(-1)).squeeze(-1)
        term_lambda1 = box1_A_p1 - box1_b
        K[:, 2*D:2*D+self.num_1_points] = lambda_1_tilde * term_lambda1

        # Lagrangian gradient for lambda2
        A2_p2 = torch.bmm(self.A_2_canonical.unsqueeze(0).expand(B, 2, 2),
                          p2_tilde.unsqueeze(-1)).squeeze(-1)
        term_lambda2 = A2_p2 - self.b_2_canonical
        K[:, 2*D+self.num_1_points:] = lambda_2_tilde * term_lambda2

        return K

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor with shape (1, 11) containing all required inputs
    return torch.rand(1, 11, dtype=torch.float32)

