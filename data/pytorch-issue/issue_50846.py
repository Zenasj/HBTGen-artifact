import timeit
import torch

torch._C._jit_override_can_fuse_on_cpu(True)                                                                                                                                                   
torch._C._debug_set_fusion_group_inlining(False)                                                                                                                                               

x = torch.randn(1024, 1024)

@torch.jit.script                                                                                                                                                                              
def script_tanh(x):
    return torch.tanh(x)

for _ in range(5):
    script_tanh(x)
    
print(timeit.timeit(stmt="torch.tanh_(x)", globals=globals(), number=1000))                                                                                                                    
print(timeit.timeit(stmt="script_tanh(x)", globals=globals(), number=1000))

2.983580728061497
15.662288276012987

from caffe2.python import core
core.GlobalInit(["python", "--torch_jit_llvm_use_fast_intrinsics", "1"])