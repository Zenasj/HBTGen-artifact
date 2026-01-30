import torch.nn as nn

import logging

import torch
from torch._dynamo import config, explain

config.verbose = True
config.log_level = logging.DEBUG

config.repro_after = "dynamo"
config.repro_level = 3

config.output_code = True
config.output_graph_code = True
config.print_graph_breaks = True


class CastToFloat(torch.nn.Module):

    def __init__(self, mod):
        super(CastToFloat, self).__init__()
        self.mod = mod

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            ret = self.mod.forward(x.to(torch.float32)).to(x.dtype)
        return ret

embedding = torch.randn(20, 5, 10).int().cuda()

layer_norm = torch.nn.LayerNorm(10).cuda()
model = CastToFloat(layer_norm)

##### fails
explain(model, embedding)


##### Also fails
compiled = torch.compile(model)
compiled(embedding)