import torch

torch.distributions.LKJCholesky.log_prob

torch.distributions.LKJCholesky(dim=2, concentration=1).log_prob(torch.eye(2).to("cuda"))

order = torch.arange(2, self.dim + 1)

order = torch.arange(2, self.dim + 1).to(device)

py
torch.tensor(0.0) + torch.eye(2).to("cuda")  # errors
torch.tensor(0.0, device="cuda") + torch.eye(2).to("cuda")  # succeeds

py
torch.distributions.LKJCholesky(
    dim=2, concentration=torch.tensor(1., device="cuda")
).log_prob(torch.eye(2).to("cuda"))

torch.distributions.LKJCholesky(
    dim=2, concentration=torch.tensor(1., device="cuda")
).log_prob(torch.eye(2).to("cuda"))

torch.distributions.LKJCholesky(
    dim=torch.tensor(2, device="cuda"), concentration=torch.tensor(1., device="cuda")
).log_prob(torch.eye(2).to("cuda"))