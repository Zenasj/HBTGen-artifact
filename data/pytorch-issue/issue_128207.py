import torch.nn as nn

with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
    scaled_dot_product_attention(...)
    ...

import torch
print("cuDNN version:", torch.backends.cudnn.version())

# will show: cuDNN version: 8907

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with torch.nn.attention.sdpa_kernel([torch.backends.cuda.SDPBackend.CUDNN_ATTENTION]):
        out = F.scaled_dot_product_attention(query,key,value)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                      aten::scaled_dot_product_attention         0.36%       1.198ms       100.00%     337.316ms     337.316ms       0.000us         0.00%      32.736us      32.736us             1  
#               aten::_scaled_dot_product_cudnn_attention        99.13%     334.405ms        99.64%     336.118ms     336.118ms      32.736us       100.00%      32.736us      32.736us             1  
# cudnn_generated_fort_native_sdpa_sm80_knob_6_128x64x...         0.00%       0.000us         0.00%       0.000us       0.000us      30.592us        93.45%      30.592us      30.592us             1  
# ...