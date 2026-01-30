import random

def type(self: T, dst_type: Union[dtype, str]) -> T:
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """

import torch
import numpy as np

x = torch.rand(5,5)
y = np.random.rand(5,5)

# conversion using the relevant dtype works
x.type(torch.float16)
y.astype(np.float16)

# np supports also dtype passed as strings
y.astype("float16")

# however, torch does not
x.type("float16")

# also this does not work
x.type("torch.float16")

torch.arange(10).type(getattr(torch, "float32"))
# returns:  tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])