import torch

from torch.utils._pytree import register_pytree_node
register_pytree_node(
    type(inputs),
    lambda d: (list(d.values()), list(d.keys())),
    lambda values, keys: type(inputs)(zip(keys, values)),
)