import torch.nn as nn

import torch

module = torch.nn.Linear(1024, 1024)
module_opt = torch.compile(module)

module_opt.eval()

print(f"{module.training=}")  # False
print(f"{module_opt.training=}")  # True
print(f"{module_opt._orig_mod.training=}")  # False

module_opt.train()

print(f"{module.training=}")  # True
print(f"{module_opt.training=}")  # True
print(f"{module_opt._orig_mod.training=}")  # True

module.training=False
module_opt.training=False
module_opt._orig_mod.training=False
module.training=True
module_opt.training=True
module_opt._orig_mod.training=True