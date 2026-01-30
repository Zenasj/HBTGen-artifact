import random

torch.einsum("xijk, bilm -> bjklm", torch.randn(1, 3, 24, 20), torch.randn(5, 3, 24, 20))

import numpy as np
import torch
np.__version__, torch.__version__
#('1.16.3', '1.1.0')

M = np.random.randn(2,2,3)
N = np.random.randn(3,3)
np.einsum('...ij, ...j->...i', N, M)
#OK

torch.einsum('...ij, ...j->...i', torch.from_numpy(N), torch.from_numpy(M))
#~/miniconda3/envs/abg3/lib/python3.7/site-packages/torch/functional.py in einsum(equation, *operands)
#    209         # the old interface of passing the operands as one list argument
#    210         operands = operands[0]
#--> 211     return torch._C._VariableFunctions.einsum(equation, operands)
# RuntimeError: ellipsis must represent 0 dimensions in all terms