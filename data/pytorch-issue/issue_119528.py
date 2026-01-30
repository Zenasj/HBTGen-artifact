import torch
import torch.nn as nn

torch._dynamo.export(torch.ops.aten.scaled_dot_product_attention, ...)(*inputs,)

torch.export

nn.Module

trace_rules.check