import torch.nn as nn

import torch
from torch.nn import Linear
from torch.autograd import Variable
from torch.optim import AdamW
from torch.utils import hooks

# nn.Module should have training attribute
module = Linear(10, 20)
module.training

# torch should have dtype bfloat16
tensor2 = torch.tensor([1,2,3], dtype=torch.bfloat16)

# torch.Tensor.cuda should accept int or str value
torch.randn(5).cuda(1)
torch.tensor(5).cuda('cuda:0')

# optimizer should have default attribute
module = Linear(10, 20)
print(AdamW(module.weight).default)

# torch.Tensor should have these boolean attributes
torch.tensor([1]).is_sparse
torch.tensor([1]).is_quantized
torch.tensor([1]).is_mkldnn

# Size class should tuple of int
a, b = torch.tensor([[1,2,3]]).size()

# check modules can be accessed
torch.nn.parallel
torch.autograd.profiler
torch.multiprocessing
torch.sparse
torch.onnx
torch.jit
torch.hub
torch.random
torch.distributions
torch.quantization
torch.__config__
torch.__future__

torch.ops
torch.classes

# Variable class's constructor should return Tensor
def fn_to_test_variable(t: torch.Tensor):
    return None

v = Variable(torch.tensor(1))
fn_to_test_variable(v)

# check RemovableHandle attributes can be accessed
handle = hooks.RemovableHandle({})
handle.id
handle.next_id

# check torch function hints
torch.is_grad_enabled()