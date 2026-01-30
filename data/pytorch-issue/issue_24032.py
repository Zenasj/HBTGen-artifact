import torch.nn as nn

import torch
import torch.nn.functional as F

@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training) :
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = F.dropout(x, p=prob, training=is_training)
    out = residual + out
    return out

@torch.jit.script
def jit_dropout_add_const(x, residual, prob) :
    # type: (Tensor, Tensor, float) -> Tensor
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


inputs = torch.ones(5, 5 , dtype=torch.float16, device=torch.device("cuda:0"), requires_grad=True)
residuals = torch.ones(5, 5 , dtype=torch.float16, device=torch.device("cuda:0"))

output_bad = jit_dropout_add(inputs, residuals, 0.1, True)
output_good = jit_dropout_add_const(inputs, residuals, 0.1)