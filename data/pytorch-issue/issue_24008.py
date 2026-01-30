[0.32401383699834696, 0.0002588090028439183, 0.00021725300030084327, 0.0002125649989466183, 0.00021101400125189684]

import time
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

mu = torch.ones(2, device='cuda')
sigma = torch.eye(2, device='cuda')
dist = MultivariateNormal(loc=mu, scale_tril=sigma)

runs = 5
times = []
X = torch.rand(1, 2, device='cuda')
for _ in range(runs):
    torch.cuda.synchronize()
    st = time.perf_counter()
    prob = dist.log_prob(X)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - st)
print(times)