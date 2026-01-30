import torch.nn as nn

import torch

self = torch.tensor([[[[0.0396193 + 1.5585054j, 0.5038033 - 1.3928472j],
                       [1.1071061 + 1.0378395j, 0.0687875 - 0.1666800j]],

                      [[-0.9338380 - 1.0284885j, 0.2591278 + 0.5482853j],
                       [0.5984055 - 0.5939694j, 0.6268274 - 1.2067362j]]]], dtype=torch.complex64)

self_cuda = self.cuda()

module = torch.nn.Tanh()
module.to(torch.complex64)
result_cpu = module(self)

module_cuda = torch.nn.Tanh()
module_cuda.to(torch.complex64)
result_gpu = module_cuda(self_cuda)

print("CPU result:\n", result_cpu)
print("GPU result:\n", result_gpu)

inconsistent = not torch.allclose(result_cpu, result_gpu.cpu(), atol=1e-05, rtol=1e-06)
print(f"inconsistency with atol=1e-05 and rtol=1e-06: {inconsistent}")