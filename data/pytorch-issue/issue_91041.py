import torch

import functorch, torch
functorch.vmap(lambda x: x.sum())(torch.tensor([10]))

import functorch, torch
functorch.vmap(lambda x: x.sum())(torch.tensor([10, 10]))