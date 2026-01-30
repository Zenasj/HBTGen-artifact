import torch.nn as nn

import torch
import torch.utils.collect_env
torch.utils.collect_env.main()

class MyModule(torch.nn.Module):
    def forward(self):
#         return do_computation()
# NOTE: This is how Jupyter comments things out: ^^^
        return torch.Tensor([])

m = MyModule()
print(m())
torch.jit.script(m)