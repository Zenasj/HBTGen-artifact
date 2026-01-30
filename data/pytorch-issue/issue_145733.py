import torchao.prototype.smoothquant
import torch
from copy import deepcopy

a = torchao.prototype.smoothquant.api._ActQuantizer(torch.int).static_quantize
b = deepcopy(a)
assert b == a