import torch.nn as nn

import torch

m = torch.nn.ChannelShuffle(2)
inp = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)

exp = torch.export.export(m, (inp,))
exp = exp.run_decompositions()
print(exp.graph_module.code)