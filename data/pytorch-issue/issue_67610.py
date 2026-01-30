import torch.nn as nn

import torch
import torchvision.models as models
import torch.optim as optim
from torch.nn.utils.memory_format import convert_conv2d_weight_memory_format

import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

device = 'lazy'
use_ts = False
if use_ts:
  device = 'cuda'

dtype = torch.float32

def model(x, w, b, rm, rv):
  o, m, v = torch.native_batch_norm(x, w, b, rm, rv, True, 0.1, 1e-5)
  #o = w + b + rm + rv
  #o = x + o.unsqueeze(-1).unsqueeze(-1)
  return torch.relu(o)

x = torch.randn(32, 64, 122, 122).to(dtype=dtype).to(device=device)
w = torch.randn(64).to(dtype=dtype).to(device=device)
b = torch.randn(64).to(dtype=dtype).to(device=device)
rm = torch.randn(64).to(dtype=dtype).to(device=device)
rv = torch.randn(64).to(dtype=dtype).to(device=device)

if use_ts:
  model = torch.jit.script(model)

for n_iter in range(8):
  pred = model(x, w, b, rm, rv)
  ltm.mark_step()
  lazy_tensor_core._LAZYC._ltc_wait_device_ops(devices=[])

print(metrics.metrics_report())