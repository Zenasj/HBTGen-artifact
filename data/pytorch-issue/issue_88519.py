r"""Performs dense-dense matrix multiplication according to segments along
    the first dimension of :obj:`inputs` as given by :obj:`ptr`, utilizing
    dedicated kernels that effectively parallelize over groups.

    .. code-block:: python

        inputs = torch.randn(8, 16)
        ptr = torch.tensor([0, 5, 8])
        other = torch.randn(2, 16, 32)

        out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
        assert out.size() == (8, 32)
        assert out[0:5] == inputs[0:5] @ other[0]
        assert out[5:8] == inputs[5:8] @ other[1]

    Args:
        input (torch.Tensor): The left operand 2D matrix of shape
            :obj:`[N, K]`.
        ptr (torch.Tensor): Compressed vector of shape :obj:`[B + 1]`, holding
            the boundaries of segments.
        other (torch.Tensor): The right operand 3D tensor of shape
            :obj:`[B, K, M]`.

    Returns:
        torch.Tensor: The 2D output matrix of shape :obj:`[N, M]`.
    """

import torch
import time
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
times = []
num_nodes_per_type = 10000
n_feats = 128
out_feats = 64

def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs) # warmup
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


for num_types in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    inputs = torch.randn((num_nodes_per_type * num_types, n_feats)).cuda()
    ptr = list(range(0, (num_types + 1) * num_nodes_per_type, num_nodes_per_type))
    nt = torch.nested.nested_tensor([inputs[ptr[i]:ptr[i+1]] for i in range(len(ptr) - 1)], device='cuda')
    other = torch.randn((num_types, n_feats, out_feats), requires_grad=True).cuda()
    nt_other = torch.nested.nested_tensor(list(other.unbind()), device='cuda')
    time = benchmark_torch_function(1, torch.bmm, nt, nt_other)
    print(f"num_types: {num_types} time: {time}")