import random

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as distf
import torch.nn as nn


# Basic process setup
# torch.set_printoptions(linewidth=150, precision=12)
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)  # numpy random seed

rank = int(os.getenv("RANK"))
local_rank = int(os.getenv("LOCAL_RANK"))
world_size = int(os.getenv("WORLD_SIZE"))
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group("nccl")


def apply_embedding_tp(par_mod: nn.Embedding, mod: nn.Embedding, world_size, rank):
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = mod.embedding_dim // world_size
    with torch.no_grad():
        par_mod.weight.copy_(torch.split(mod.weight, output_size_per_partition, dim=1)[rank])
    # print(f"For rank {rank}, we have the following weights: Base weight {mod.weight} bias {mod.bias}; Par weight {par_mod.weight}, bias {par_mod.bias}")


def _all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    world_size = dist.get_world_size()

    if world_size == 1:
        return input_
    # print("Graph break 1")
    out = distf.all_reduce(input_, "sum", list(range(world_size)))
    # print("Graph break 2")
    return out


def _all_gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    world_size = dist.get_world_size()

    if world_size == 1:
        return input_

    last_dim = input_.dim() - 1
    # print("Graph break 1")
    res = distf.all_gather_tensor(input_, 0, dist.distributed_c10d._world._default_pg)
    # print("Graph break 2")
    out = torch.cat(torch.chunk(res, world_size, dim=0), dim=last_dim)

    return out


def _split(input_: torch.Tensor, rank=0, world_size=1) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    if world_size == 1:
        return input_

    # Split along last dimension.
    # Get the size and dimension.
    last_dim = input_.dim() - 1
    last_dim_size = input_.size()[last_dim] // world_size
    # Split.
    input_list = torch.split(input_, last_dim_size, dim=last_dim)

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _all_reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _all_reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _AllGatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _all_gather(input_)

    @staticmethod
    def forward(ctx, input_):
        ctx.rank = rank
        ctx.world_size = dist.get_world_size()
        return _all_gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.rank, ctx.world_size)


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def all_gather_from_tensor_model_parallel_region(input_):
    return _AllGatherFromModelParallelRegion.apply(input_)


class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.emb(x)


class TPModel(Model):
    def __init__(self, vocab_size, emb_dim, rank, world_size, *args, **kwargs) -> None:
        super().__init__(vocab_size, emb_dim // world_size, *args, **kwargs)
        self.rank = rank
        self.world_size = world_size

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x)
        y_par = self.emb(x_par)
        return all_gather_from_tensor_model_parallel_region(y_par)


model = Model(32, 32).to(device)
tp_model = TPModel(32, 32, rank, world_size).to(device)

# Copy weights
apply_embedding_tp(tp_model.emb, model.emb, world_size, rank)

tp_model = torch.compile(tp_model)

l_seq = 10
l_batch = 1

# Simulate tokenization on only one process
if rank == 0:
    inp = torch.randint(0, 32, (l_batch, l_seq), dtype=torch.long).to(device)
else:
    inp = torch.zeros([l_batch, l_seq], dtype=torch.long).to(device)
torch.distributed.broadcast(inp, 0)

# See issue #107824 for why no_grad() is needed
with torch.no_grad():
    out = model(inp)
    tp_out = tp_model(inp)
    torch.cuda.synchronize()
    print(f"Error report rank {rank}: {out-tp_out};  Avg abs error: {(out-tp_out).abs().mean()}")

# torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS repro.py

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as _functional_collectives
import torch._inductor.config
import torch.nn as nn

rank = int(os.getenv("RANK"))
local_rank = int(os.getenv("LOCAL_RANK"))
world_size = int(os.getenv("WORLD_SIZE"))
device = torch.device("cuda", local_rank)
tag = ""
ranks = list(range(world_size))
group_size = world_size
torch.cuda.set_device(device)
dist.init_process_group("nccl")

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = torch.nn.Embedding(4, 4)

    def forward(self, x):
        y = self.emb(x)
        last_dim = y.dim() - 1
        res = _functional_collectives.all_gather_tensor(y, 0, ranks, tag)
        out = torch.cat(torch.chunk(res, world_size, dim=0), dim=last_dim)
        return out

torch._inductor.config.allow_buffer_reuse = True

tp_model = Model().cuda()
tp_model_compiled = torch.compile(tp_model)

inp = torch.tensor([[2, 1, 3, 0]], dtype=torch.long, device="cuda")

# See issue #107824 for why no_grad() is needed
with torch.no_grad():
    tp_out = tp_model(inp)
    tp_compiled_out = tp_model_compiled(inp)
    assert torch.allclose(tp_compiled_out, tp_out)

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (1, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 10, 4), (40, 4, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [emb], Original ATen: [aten.embedding]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_embedding_0.run(arg1_1, arg0_1, buf0, 40, grid=grid(40), stream=stream0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided((2, 10, 4), (40, 4, 1), device='cuda', dtype=torch.float32)
        # collective out buffer buf1
        buf2_pg = c10d._find_or_create_pg_by_ranks_and_tag('ptd:0', [0, 1], 2)
        buf2_inputs = [buf0]
        buf2 = [buf1]
        buf2_work = dist.all_gather_into_tensor(buf2[0], buf2_inputs[0], async_op=True, group=buf2_pg)
        fun_col_impl._register_tensor_work(buf2, buf2_work)
        buf3 = buf2[0]
        del buf0
        buf3 = _wait_tensor(buf3)
        buf4 = buf3
        buf7 = reinterpret_tensor(buf1, (1, 10, 8), (80, 8, 1), 0); del buf1  # reuse
        buf5 = reinterpret_tensor(buf7, (1, 10, 4), (80, 8, 1), 0)  # alias
        # Source Nodes: [cat_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(buf4, buf5, 40, grid=grid(40), stream=stream0)
        buf6 = reinterpret_tensor(buf7, (1, 10, 4), (80, 8, 1), 4)  # alias
        # Source Nodes: [cat_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_2.run(buf4, buf6, 40, grid=grid(40), stream=stream0)
        del buf3
        del buf4
        return (buf7, )

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (1, 10), (10, 1))
    with torch.cuda._DeviceGuard(1):
        torch.cuda.set_device(1) # no-op to ensure context
        buf0 = empty_strided((1, 10, 4), (40, 4, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [emb], Original ATen: [aten.embedding]
        stream1 = get_cuda_stream(1)
        triton_poi_fused_embedding_0.run(arg1_1, arg0_1, buf0, 40, grid=grid(40), stream=stream1)
        del arg0_1
        del arg1_1
        buf1 = empty_strided((2, 10, 4), (40, 4, 1), device='cuda', dtype=torch.float32)
        # collective out buffer buf1
        buf2_pg = c10d._find_or_create_pg_by_ranks_and_tag('ptd:0', [0, 1], 2)
        buf2_inputs = [buf0]
        buf2 = [buf1]
        buf2_work = dist.all_gather_into_tensor(buf2[0], buf2_inputs[0], async_op=True, group=buf2_pg)
        fun_col_impl._register_tensor_work(buf2, buf2_work)
        del buf1
        buf3 = buf2[0]
        del buf0
        buf3 = _wait_tensor(buf3)
        buf4 = buf3
        buf7 = empty_strided((1, 10, 8), (80, 8, 1), device='cuda', dtype=torch.float32)
        buf5 = reinterpret_tensor(buf7, (1, 10, 4), (80, 8, 1), 0)  # alias
        # Source Nodes: [cat_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_1.run(buf4, buf5, 40, grid=grid(40), stream=stream1)
        buf6 = reinterpret_tensor(buf7, (1, 10, 4), (80, 8, 1), 4)  # alias
        # Source Nodes: [cat_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_2.run(buf4, buf6, 40, grid=grid(40), stream=stream1)
        del buf3
        del buf4
        return (buf7, )