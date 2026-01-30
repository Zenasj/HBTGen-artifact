import torch.nn as nn

import math
import torch


inp = 3.
target = 5.

nll_poisson_math = inp - target * math.log(inp) + math.log(math.factorial(target))
nll_poisson_stirling_math = inp - target * math.log(inp) + target * math.log(target) - target + 0.5 * math.log(2 * math.pi * target)

nll_poisson_loss = torch.nn.PoissonNLLLoss(log_input=False, 
                        full=True,
                        size_average=None,
                        eps=0.,
                        reduce=None,
                        reduction='none')

inpTensor, targetTensor = torch.as_tensor([inp]), torch.as_tensor([target])
nll_poisson_pytorch = nll_poisson_loss(log_input=inpTensor, target=targetTensor)

print(nll_poisson_math, nll_poisson_stirling_math, nll_poisson_pytorch.item())