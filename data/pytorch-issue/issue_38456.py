import torch

(
    torch.distributions.Binomial(
        total_count=torch.tensor(1.0).cuda(), probs=torch.tensor(0.9).cuda()
    ).sample(torch.Size((1000000000,))) >= 0
).all()