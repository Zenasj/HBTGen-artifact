"""
$ python repro.py
torch==2.5.0.dev20240818+cu121
/scratch/slurm_tmpdir/620813/env/lib/python3.11/site-packages/torch/autograd/graph.py:817: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at ../aten/src/ATen/cuda/CublasHandlePool.cpp:135.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/scratch/slurm_tmpdir/620813/env/lib/python3.11/site-packages/torch/autograd/graph.py:817: UserWarning: cuDNN SDPA backward got grad_output.strides() != output.strides(), attempting to materialize a grad_output with matching strides... (Triggered internally at ../aten/src/ATen/native/cudnn/MHA.cpp:667.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Traceback (most recent call last):
  File "/storage/home/user/work/repro.py", line 38, in <module>
    y.backward(y)
  File "/scratch/slurm_tmpdir/620813/env/lib/python3.11/site-packages/torch/_tensor.py", line 522, in backward
    torch.autograd.backward(
  File "/scratch/slurm_tmpdir/620813/env/lib/python3.11/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/scratch/slurm_tmpdir/620813/env/lib/python3.11/site-packages/torch/autograd/graph.py", line 817, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: std::equal( dO_.strides().begin(), dO_.strides().end(), o.strides().begin()) INTERNAL ASSERT FAILED at "../aten/src/ATen/native/cudnn/MHA.cpp":676, please report a bug to PyTorch. cuDNN SDPA expected grad_output.strides() == output.strides(), the previous step probably failed to materialize a grad_output with matching strides...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

d = 64
seqlen = 4
bs = 2

class AttentionSDPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.wo = nn.Linear(d, d)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.wo(
            F.scaled_dot_product_attention(x, x, x)
            .transpose(1, 2)
            .reshape([x.shape[0], seqlen, -1])
        )


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.attnSDPA = AttentionSDPA()

    def forward(self, x):
        x = self.q_proj(x).reshape([bs, seqlen, d // 64, 64])
        x = self.attnSDPA(x)
        return x

print(f"torch=={torch.__version__}")
kw = dict(device="cuda", dtype=torch.float16)
m = Model()
m.to(**kw)
x = torch.randn([bs, seqlen, d], **kw)
y = m(x)
# RuntimeError: std::equal( dO_.strides().begin(), dO_.strides().end(), o.strides().begin()) INTERNAL ASSERT FAILED
y.backward(y)