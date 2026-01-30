import torch

py
# You don't need to import RRef.

# type: (RRef[Tensor]) -> RRef[Tensor]

py
from torch._jit_internal import RRef

rref_module : RRef[MyModuleInterface]

py
from torch.rpc.distritubed import RRef

rref_module : RRef[MyModuleInterface]

py
from torch._jit_internal import RRef

rref_module : RRef[MyModuleInterface]