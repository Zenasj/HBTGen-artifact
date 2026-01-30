import torch.nn as nn

import torch
import transformers
from transformers import file_utils  # Crucial, otherwise dynamo fails.
from transformers.modeling_outputs import CausalLMOutputWithPast

class Model(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        return CausalLMOutputWithPast(loss=None, logits=x)[0]

model = Model()
x = torch.randn(1, 1, 1, 1)
eo = torch._dynamo.export(Model(), aten_graph=True)(x)
eo.graph_module.print_readable()
print("dynamo output: ", eo.graph_module(x))
print("eager output: ", model(x))