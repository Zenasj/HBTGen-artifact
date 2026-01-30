import numpy as np
import torch

evals = torch.logspace(-4, -3, 100, dtype=torch.double)
cov = torch.diag(evals)

print(torch.logdet(cov))
print(torch.sum(torch.log(torch.svd(cov)[1])))
print(torch.log(torch.prod(torch.svd(cov)[1])))
print(np.linalg.slogdet(cov)[1])