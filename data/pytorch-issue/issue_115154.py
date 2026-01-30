import torch.nn as nn

import torch
import torch.nn.functional as F
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, W: torch.Tensor, B: torch.Tensor):
        ctx.wcsplt = torch.ops.aten._cslt_compress(W)
        ctx.wcspltT = torch.ops.aten._cslt_compress(W.t().contiguous())
        ctx.save_for_backward(x)
        return torch.ops.aten._cslt_sparse_mm(ctx.wcsplt, dense_B=x.t(), bias=B, transpose_result=True)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x = ctx.saved_tensors[0]
        return (
            torch.ops.aten._cslt_sparse_mm(ctx.wcsplt, grad_out.t(), transpose_result=True),
            x.t() @ grad_out,
            None
        )

m, k, n = 16384, 4096, 512
stride = 512
a = torch.randn([m, k], device="cuda", dtype=torch.float16, requires_grad=True)
W = torch.randn([n, k], device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.zeros([n], device="cuda", dtype=torch.float16, requires_grad=True)
W_pruned = W.clone()
for i in range(n):
    for j in range(k//stride):
        W_pruned[i][j*stride:j*stride+stride] *= torch.tensor([1, 0, 1, 0] * (stride // 4), device=W.device)

# Timing dense operation
start_time_dense = torch.cuda.Event(enable_timing=True)
end_time_dense = torch.cuda.Event(enable_timing=True)
start_time_dense.record()
out_dense = F.linear(a, W_pruned, B)
end_time_dense.record()
torch.cuda.synchronize()
time_cost_dense = start_time_dense.elapsed_time(end_time_dense)

# Timing sparse operation
start_time_sparse = torch.cuda.Event(enable_timing=True)
end_time_sparse = torch.cuda.Event(enable_timing=True)
start_time_sparse.record()
out_sparse = Linear.apply(a, W_pruned, B)
end_time_sparse.record()
torch.cuda.synchronize()
time_cost_sparse = start_time_sparse.elapsed_time(end_time_sparse)

print("Dense time:", time_cost_dense, "Sparse time:", time_cost_sparse, "Ratio:", time_cost_dense / time_cost_sparse)
assert torch.allclose(out_sparse, out_dense, atol=1e-3), "The outputs of the sparse and dense operations are not close enough."

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
SparseSemiStructuredTensor._FORCE_CUTLASS = False
import numpy as np
torch.manual_seed(114)
torch.cuda.manual_seed(114)

# mask Linear weight to be 2:4 sparse
mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 2560)).cuda().bool()
linear = torch.nn.Linear(10240, 3072).half().cuda().eval()
linear.weight = torch.nn.Parameter((mask * linear.weight * 128).to(torch.int8), requires_grad=False)
# bias = torch.randint(low=0, high=10, size=(3072,), dtype=torch.int32).cuda()
bias = torch.randn(3072, dtype=torch.int32).cuda()

x = (torch.rand(3072, 10240).half() * 127).cuda().to(torch.int8)
print("x: ",x)
print("linear.weight: ",linear.weight)
with torch.inference_mode():
    x_ = x.to(torch.float16)
    weight_ = linear.weight.to(torch.float16)
    bias_ = bias.to(torch.float16)
    dense_output = torch.nn.functional.linear(x_, weight_, bias_)
    dense_output_int8 = torch.from_numpy(np.clip(dense_output.cpu().numpy(), -128, 127)).type(torch.int8).cuda()

    dense_t = Timer(stmt="torch.nn.functional.linear(x, weight, bias)",
                    globals={"x": x_, "weight": weight_, "bias": bias_}).blocked_autorange().median * 1e3
    
    # accelerate via SparseSemiStructuredTensor
    linear.weight_int8 = torch.nn.Parameter(to_sparse_semi_structured(linear.weight), requires_grad=False)
    sparse_output_int8 = torch.nn.functional.linear(x, linear.weight_int8, bias)
    sparse_t_int8 = Timer(stmt="torch.nn.functional.linear(x, weight, bias)",
                    globals={"x": x, "weight": linear.weight_int8, "bias": bias}).blocked_autorange().median * 1e3

    weight_ = torch.nn.Parameter(to_sparse_semi_structured(weight_), requires_grad=False)
    sparse_output_fp16 = torch.nn.functional.linear(x_, weight_, bias_)
    sparse_output_fp16_int8 = torch.from_numpy(np.clip(sparse_output_fp16.cpu().numpy(), -128, 127)).type(torch.int8).cuda()
    sparse_t_fp16 = Timer(stmt="torch.nn.functional.linear(x, weight, bias)",
                    globals={"x": x_, "weight": weight_, "bias": bias_}).blocked_autorange().median * 1e3
    print("dense: ", dense_output_int8)
    print("sparse_int8: ", sparse_output_int8)
    print("sparse_fp16: ", sparse_output_fp16_int8)
    
    difference = torch.abs(sparse_output_int8 - dense_output_int8)

    difference_threshold = 1
    mismatched_positions = torch.nonzero(difference > difference_threshold)

    for position in mismatched_positions:
        pos = tuple(position.tolist())
        sparse_val = sparse_output_int8[pos]
        dense_val = dense_output_int8[pos]
        print(f"Mismatch at position {pos}: sparse={sparse_val}, dense={dense_val}")


    assert torch.allclose(sparse_output_int8, dense_output_int8, atol=2)
    assert torch.allclose(sparse_output_fp16_int8, dense_output_int8, atol=2)
    assert torch.allclose(sparse_output_int8, sparse_output_fp16_int8, atol=0)
    print(f"Sparse_fp16: {sparse_t_fp16:.3f}ms Sparse_int8: {sparse_t_int8:.3f}ms | Speedup: {(sparse_t_fp16 / sparse_t_int8):.3f}x")
    print(f"Dense: {dense_t:.3f}ms Sparse_int8: {sparse_t_int8:.3f}ms | Speedup: {(dense_t / sparse_t_int8):.3f}x")
    print(f"Dense: {dense_t:.3f}ms Sparse_fp16: {sparse_t_fp16:.3f}ms | Speedup: {(dense_t / sparse_t_fp16):.3f}x")