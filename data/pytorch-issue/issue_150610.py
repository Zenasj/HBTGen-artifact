import random

import torch
import numpy as np
input_data = np.random.rand(10, 10)
input_tensor = torch.from_numpy(input_data)
input_tensor = torch.autograd.Variable(input_tensor, requires_grad=False)
optimizer = torch.optim.Adam([{'params': input_tensor, 'weight_decay': 0.01}])
# optimizer = torch.optim.AdamW([{'params': input_tensor, 'weight_decay': 0.01}])
# optimizer = torch.optim.RAdam([{'params': input_tensor, 'weight_decay': 0.01}])
optimizer.step()

class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized,
            which should have grad available.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """