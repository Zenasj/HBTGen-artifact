import torch.nn as nn

module = torch.nn.Linear(20, 40)
module = module.to(torch.bfloat16)
module = module.to("cuda")
nn.utils.weight_norm(module, "weight")