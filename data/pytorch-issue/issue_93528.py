py
# python3 benchmark.py

import torch

import triton
import triton.language as tl
import time

BLOCK_M = 128
BLOCK_N = 128

@triton.jit
def _load_and_op(A, B, M, N,
                 stride_m, stride_n,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 OP_MODE: tl.constexpr, CONTENDED: tl.constexpr):
    if CONTENDED:
      m = tl.arange(0, BLOCK_M)
      n = tl.arange(0, BLOCK_N)
    else:
      pid_m = tl.program_id(0)
      pid_n = tl.program_id(1)
      m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
      n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (m < M)[:, None] & (n < N)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=A.dtype.element_ty)
    A = A + (m[:, None] * stride_m + n[None, :] * stride_n)
    B = B + (m[:, None] * stride_m + n[None, :] * stride_n)

    b = tl.load(B, mask=mask, other=0.)
    if OP_MODE == 0:
      # Add-store
      a = tl.load(A, mask=mask, other=0.)
      b = a + b
      tl.store(A, b, mask=mask)
    elif OP_MODE == 1:
      # Atomic Add
      tl.atomic_add(A, b, mask=mask)
    elif OP_MODE == 2:
      # Noop
      x = 0


def load_and_op(A, B, M, N, np_m, np_n, op_mode, contended, check=False):
  grid = lambda META: (np_m, np_n)
  if check:
    if (op_mode == 0 and contended == False) or (op_mode == 1 and contended == False):
      ans = A + B
      _load_and_op[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M, BLOCK_N, op_mode, contended)
      assert(torch.allclose(ans, A))
    elif op_mode == 1 and contended == True:
      ans = A + B
      for i in range(np_m * np_n - 1):
        ans += B
      _load_and_op[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M, BLOCK_N, op_mode, contended)
      assert(torch.allclose(ans, A))
    else:
      # No way to check it.
      _load_and_op[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M, BLOCK_N, op_mode, contended)
  else:
    # Run without check
    _load_and_op[grid](A, B, M, N, A.stride(0), A.stride(1), BLOCK_M, BLOCK_N, op_mode, contended)


def benchmark_func(f, args):
    iters = 1000
    num_warmups = 100

    output = f(*args, check=True)
    for _ in range(num_warmups):
      output = f(*args)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(iters)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        start_event[i].record()
        output = f(*args)
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    )
    elapsed_time = torch.mean(times).item() * 1.0e-3 # ms -> s
    # pyre-fixme[61]: `output` is undefined, or not always defined.
    return float(elapsed_time), output


print("dtype\tnp_n\tnp_m\top_mode\tcontended\tt_us")
for dtype in [torch.float, torch.half]:
  np_n = 1
  for (op_mode, contended) in [(0, False), (1, True), (1, False), (2, False)]:
    for np_m in [2 ** x for x in range(0, 16)]:
      if contended:
        M = BLOCK_M
        N = BLOCK_N
      else:
        M = BLOCK_M * np_m
        N = BLOCK_N * np_n
      A = torch.rand((M, N), device="cuda", dtype=dtype)
      B = torch.rand((M, N), device="cuda", dtype=dtype)
      t, _ = benchmark_func(load_and_op, (A, B, M, N, np_m, np_n, op_mode, contended))
      print(f"{dtype}\t{np_n}\t{np_m}\t{op_mode}\t{contended}\t{t * 1.0e6}")

py
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

py
# python3 bench2.py

import torch
import torch._dynamo
import time

batch_size = 100
tgt_dim = 1000000
src_dim = 300000
max_spread_pow = 10

def torch_kernel(src, tgt, index):
    tgt.scatter_add_(1, index, src)

def torch_kernel2(src, tgt, index):
    tgt = tgt.scatter_add(1, index, src)

@torch._dynamo.optimize("inductor")
def dynamo_kernel(src, tgt, index):
    tgt = tgt.scatter_add(1, index, src)

for inductor in [0, 1, 2]:
    fp32_results = []
    fp16_results = []
    for spread_pow in range(1, max_spread_pow):
        for precision in ["fp32", "fp16"]:
                dtype = torch.float32 if precision == "fp32" else torch.float16
                tgt = torch.zeros(batch_size, tgt_dim).to(dtype).cuda()
                src = torch.rand(batch_size, src_dim).to(dtype).cuda()
                spread = 0.5 ** spread_pow
                index = torch.randint(
                    int(tgt_dim / 2 - tgt_dim * spread),
                    int(tgt_dim / 2 + tgt_dim * spread),
                    (batch_size, src_dim)
                ).cuda()

                num_reps = 10
                num_warmups = 10
                t_mean = 0.0
                for i in range(num_reps + num_warmups):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    if inductor == 0:
                        torch_kernel(src, tgt, index)
                    elif inductor == 1:
                        torch_kernel2(src, tgt, index)
                    elif inductor == 2:
                        dynamo_kernel(src, tgt, index)
                    t1 = time.time()
                    if i >= num_warmups:
                        t_mean += t1 - t0
                t_mean = t_mean / num_reps

                if precision == "fp32":
                    fp32_results.append(t_mean)
                else:
                    fp16_results.append(t_mean)

    if inductor == 0:
        inductor_desc = "eager-inplace"
    elif inductor == 1:
        inductor_desc = "eager-outplace"
    elif inductor == 2:
        inductor_desc = "inductor-outplace"

    desc = f"fp32-{inductor_desc}\t"
    desc += "\t".join([str(f) for f in fp32_results])
    desc += f"\nfp16-{inductor_desc}\t"
    desc += "\t".join([str(f) for f in fp16_results])
    print(desc)

print("\t" + ("\t".join([f"{2 * 0.5 ** spread_pow:.5f}" for spread_pow in range(1, max_spread_pow)])))

@torch._dynamo.optimize("inductor")
def dynamo_kernel(src, tgt, index):
    return tgt.scatter_add(1, index, src)

py
import torch._inductor.config
torch._inductor.config.triton.cudagraphs = False

py
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 300000)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tl.atomic_add(out_ptr0 + (tmp0 + (100000*x1) + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)

py
import torch
import torch._dynamo
import torch._inductor.config
import time

batch_size = 100
tgt_dim = 1000000
src_dim = 300000
max_spread_pow = 10
torch._inductor.config.triton.cudagraphs = False

def torch_kernel(src, tgt, index):
    tgt.scatter_add_(1, index, src)

def torch_kernel2(src, tgt, index):
    return tgt.scatter_add(1, index, src)

@torch._dynamo.optimize("inductor")
def dynamo_kernel(src, tgt, index):
    return tgt.scatter_add(1, index, src)

for inductor in [0, 1, 2]:
    fp32_results = []
    fp16_results = []
    for spread_pow in range(1, max_spread_pow):
        for precision in ["fp32", "fp16"]:
                dtype = torch.float32 if precision == "fp32" else torch.float16
                tgt = torch.zeros(batch_size, tgt_dim).to(dtype).cuda()
                src = torch.rand(batch_size, src_dim).to(dtype).cuda()
                spread = 0.5 ** spread_pow
                index = torch.randint(
                    int(tgt_dim / 2 - tgt_dim * spread),
                    int(tgt_dim / 2 + tgt_dim * spread),
                    (batch_size, src_dim)
                ).cuda()

                num_reps = 10
                num_warmups = 10
                t_mean = 0.0
                for i in range(num_reps + num_warmups):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    if inductor == 0:
                        torch_kernel(src, tgt, index)
                    elif inductor == 1:
                        torch_kernel2(src, tgt, index)
                    elif inductor == 2:
                        dynamo_kernel(src, tgt, index)
                    t1 = time.time()
                    if i >= num_warmups:
                        t_mean += t1 - t0
                t_mean = t_mean / num_reps

                if precision == "fp32":
                    fp32_results.append(t_mean)
                else:
                    fp16_results.append(t_mean)

    if inductor == 0:
        inductor_desc = "eager-inplace"
    elif inductor == 1:
        inductor_desc = "eager-outplace"
    elif inductor == 2:
        inductor_desc = "inductor-outplace"

    desc = f"fp32-{inductor_desc}\t"
    desc += "\t".join([str(f) for f in fp32_results])
    desc += f"\nfp16-{inductor_desc}\t"
    desc += "\t".join([str(f) for f in fp16_results])
    print(desc)

print("\t" + ("\t".join([f"{2 * 0.5 ** spread_pow:.5f}" for spread_pow in range(1, max_spread_pow)])))

py
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


kernel0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 300000)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tl.atomic_add(out_ptr0 + (tmp0 + (100000*x1) + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf0 = empty_strided((100, 100000), (100000, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(arg1_1, buf0, 10000000, grid=grid(10000000), stream=stream0)
    del arg1_1
    kernel1.run(arg2_1, arg0_1, buf0, 30000000, grid=grid(30000000), stream=stream0)
    del arg0_1
    del arg2_1
    return (buf0, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((100, 300000), (300000, 1), device='cuda', dtype=torch.float32)
    arg1_1 = rand_strided((100, 100000), (100000, 1), device='cuda', dtype=torch.float32)
    arg2_1 = rand_strided((100, 300000), (300000, 1), device='cuda', dtype=torch.int64)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))