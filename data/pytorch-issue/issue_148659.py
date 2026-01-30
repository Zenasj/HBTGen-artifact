import torch
from habana_frameworks.torch import hpu

device = torch.device('hpu')
nt = torch.nested.nested_tensor(
    [torch.arange(12).reshape(2, 6), torch.arange(18).reshape(3, 6)],
    dtype=torch.float,
    device=device
)
nt = nt.detach()
unbind_result = nt.unbind()
for i, tensor in enumerate(unbind_result):
    print(f"Tensor {i}: {tensor}")