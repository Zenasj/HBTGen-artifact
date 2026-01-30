import timeit
import torch

torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._debug_set_fusion_group_inlining(False)

x = torch.randn(2**18)

@torch.jit.script
def script_add(x):
    return x + x

for _ in range(5):
    script_add(x)

print(timeit.timeit(stmt="x + x", globals=globals(), number=1000))
print(timeit.timeit(stmt="script_add(x)", globals=globals(), number=1000))

0.02427208423614502
0.1031530424952507