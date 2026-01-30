import torch
from torch import Tensor

@torch.jit.script
def fit(data: Tensor):
    # Reduce all but the last dimension
    reduction_dims = [i for i in range(data.dim() - 1)]

    data_mu = torch.mean(data, dim=reduction_dims, keepdim=True)
    data_sigma = torch.std(data, dim=reduction_dims, keepdim=True)
    data_sigma = torch.where(data_sigma < 1e-12, torch.ones_like(data_sigma), data_sigma)
    return data_mu, data_sigma

print(fit.code)

@torch.jit.script
def fit(data: Tensor):
    reduction_dims = [i for i in range(data.dim() - 1)]

    data_mu = torch.mean(data, dim=reduction_dims, keepdim=True)
    data_sigma = torch.std(data, dim=reduction_dims, keepdim=True)
    data_sigma[data_sigma < 1e-12].fill_(1.0)
    return data_mu, data_sigma

print(fit.code)

@torch.jit.script
def fit(data: Tensor):
    reduction_dims = [i for i in range(data.dim() - 1)]

    data_mu = torch.mean(data, dim=reduction_dims, keepdim=True)
    data_sigma = torch.std(data, dim=reduction_dims, keepdim=True)
    data_sigma[data_sigma < 1e-12] = 1.0
    return data_mu, data_sigma

print(fit.code)

def fit(data: Tensor) -> Tuple[Tensor, Tensor]:
  reduction_dims = annotate(List[int], [])
  for i in range(torch.sub(torch.dim(data), 1)):
    _0 = torch.append(reduction_dims, i)
  data_mu = torch.mean(data, reduction_dims, True, dtype=None)
  data_sigma = torch.std(data, reduction_dims, True, True)
  _1 = torch.lt(data_sigma, 9.9999999999999998e-13)
  _2 = torch.tensor(1., dtype=ops.prim.dtype(data_sigma), device=ops.prim.device(data_sigma), requires_grad=False)
  _3 = annotate(List[Optional[Tensor]], [_1])
  _4 = torch.index_put_(data_sigma, _3, _2, False)
  return (data_mu, data_sigma)