import torch

def forward(self, theta):
        cos_theta = torch.cos(theta)

        H = torch.ones((len(theta), 2, 2), device=theta.device, dtype=theta.dtype)
        H[:,0,0] = 1.2
        H[:,0,1] = cos_theta
        H[:,1,0] = cos_theta
        H[:,1,1] = 0.01

        return torch.inverse(H)