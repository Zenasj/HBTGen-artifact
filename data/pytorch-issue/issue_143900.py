import torch.nn as nn

import argparse
import os
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from tqdm.auto import tqdm

torch._dynamo.config.inline_inbuilt_nn_modules = False
torch._dynamo.config.optimize_ddp = False


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class NonLearnableConv(nn.Module):
    def __init__(self, kernel: Tuple[int], in_channels: int):
        super().__init__()

        self.padding = (len(kernel) - 1) // 2

        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.outer(kernel)[None, None].repeat(in_channels, 1, 1, 1)

        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.dtype, self.kernel.dtype)
        return nn.functional.conv2d(x, self.kernel, groups=self.kernel.shape[0], stride=2, padding=self.padding)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    args = parser.parse_args()
    return args


def main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    model = nn.Sequential(
        nn.Sequential(nn.Conv2d(3, 64, 3, padding=1)),
        nn.Sequential(NonLearnableConv((1, 2, 2, 1), 64)),
        nn.Sequential(nn.Conv2d(64, 3, 3, padding=1)),
        nn.Sequential(NonLearnableConv((1, 2, 2, 1), 3)),
    ).to(device)

    if args.use_fsdp:
        model = FSDP(
            module=model,
            device_id=rank,
            use_orig_params=args.use_compile,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            forward_prefetch=True,
            limit_all_gathers=True,
            auto_wrap_policy=ModuleWrapPolicy({nn.Sequential}),
            mixed_precision=MixedPrecision(
                param_dtype=dtype,
                buffer_dtype=dtype,
                reduce_dtype=dtype,
            ),
        )
        loss_amp_context = torch.amp.autocast("cuda", dtype=dtype, enabled=True)
        model_amp_context = nullcontext()
        scaler = ShardedGradScaler(enabled=dtype is torch.float16)
    else:
        loss_amp_context = torch.amp.autocast("cuda", dtype=dtype, enabled=True)
        model_amp_context = loss_amp_context
        scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)

    if args.use_compile:
        print("Trying compile.")
        model.compile(mode="default", dynamic=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98))

    iterator = range(args.num_iterations)
    if rank == 0:
        iterator = tqdm(iterator, total=args.num_iterations, miniters=10)

    for _ in iterator:
        for _ in range(args.grad_accum_steps):

            x = torch.randn(args.batch_size, 3, 128, 128, device=device)
            with model_amp_context:
                out = model(x)
            with loss_amp_context:
                loss = out.mean() / args.grad_accum_steps

            loss_test = loss.clone()  # Ensure local loss is not changed by allreduce
            torch.distributed.all_reduce(loss_test)  # Check if any gpu has NaN loss
            if torch.isnan(loss_test):
                raise ValueError("NaN loss.")

            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.freeze_support()
    if world_size == 1:
        main(0, world_size, args)
    else:
        torch.multiprocessing.spawn(fn=main, args=(world_size, args), nprocs=world_size, join=True)