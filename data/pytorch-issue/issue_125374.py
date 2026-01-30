import torch
import os
import sys
import shutil

directory_path = '/tmp/torchinductor_chilli/'
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    shutil.rmtree(directory_path)
import torch._inductor.config
torch.set_default_device('cuda')

torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000

@torch.compile(mode="max-autotune-no-cudagraphs")
def f(a, b):
    return torch.mm(a, b)

try:
    for N in range(512, 1024, 16):
        print(N)
        a = torch.randn(N, N, dtype=torch.bfloat16)
        b = torch.randn(N, N, dtype=torch.bfloat16)
        f(a, b)
except Exception as e:
    print(e)
    sys.exit(1)
sys.exit(0)