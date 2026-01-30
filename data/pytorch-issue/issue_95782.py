py
import torch

torch.manual_seed(420)

x = torch.arange(start=-10, end=10, step=0.5, dtype=torch.float).cuda()
qx = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint4x2).cuda()

# RuntimeError: cuda_dispatch_ptr INTERNAL ASSERT FAILED at 
# "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/native/DispatchStub.cpp":120, 
# please report a bug to PyTorch. DispatchStub: missing CUDA kernel