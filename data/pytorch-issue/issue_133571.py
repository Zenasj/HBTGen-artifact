import torch.nn as nn

import argparse
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from tqdm.auto import tqdm

torch._dynamo.config.optimize_ddp = False

from typing import List, Optional

import torch.nn.functional as F
from einops import rearrange
from torch import nn


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialToSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).view(b, h * w, c)


class SeqToSpatial(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, n, c = x.shape
        spatial_dim = int(n**0.5)
        return x.permute(0, 2, 1).view(b, c, spatial_dim, spatial_dim)


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, d_head: int):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.d_head = d_head
        self.n_heads = self.out_dim // self.d_head
        self.d_attn = self.out_dim

        self.pre_norm = nn.LayerNorm(input_dim)
        self.qkv_proj = nn.Linear(input_dim, 3 * self.d_attn, bias=False)
        self.q_norm = nn.RMSNorm(self.d_attn, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.d_attn, eps=1e-6)
        self.to_out = nn.Linear(self.d_attn, self.out_dim)

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.pre_norm(x)
        q, k, v = self.qkv_proj(x).chunk(dim=-1, chunks=3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.n_heads)
        out = self.to_out(out)
        return out


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        mid_channels: Optional[int] = None,
        use_conv: bool = False,
        up: bool = False,
        down: bool = False,
        norm_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.mid_channels = mid_channels or self.out_channels
        self.use_conv = use_conv

        conv_block = [nn.SiLU(), nn.Conv2d(channels, self.mid_channels, 3, padding=1)]

        self.in_layers = nn.ModuleList([nn.GroupNorm(num_channels=channels, num_groups=norm_groups), *conv_block])
        self.in_layers_len = len(self.in_layers)
        self.updown = up or down

        if up:
            self.h_upd = Upsample()
            self.x_upd = Upsample()
        elif down:
            self.h_upd = Downsample()
            self.x_upd = Downsample()
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # override num group for shrinked model
        norm_groups = max(norm_groups * self.mid_channels // self.out_channels, 1)
        self.out_layers = nn.ModuleList(
            [
                nn.GroupNorm(num_channels=self.mid_channels, num_groups=norm_groups),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Conv2d(self.mid_channels, self.out_channels, 3, padding=1)),
            ]
        )
        self.out_layers_len = len(self.out_layers)

        if use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
        else:
            if self.out_channels == channels:
                self.skip_connection = nn.Identity()
            else:
                self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor):
        h = x
        for i in range(self.in_layers_len - 1):
            h = self.in_layers[i](h)
        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)
        h = self.in_layers[self.in_layers_len - 1](h)

        for i in range(self.out_layers_len):
            h = self.out_layers[i](h)
        out = self.skip_connection(x) + h
        return out


class UNet(nn.Module):
    def __init__(self, in_dim: int, channels: List[int], attns: List[int], middle_attns: int = 0):
        super().__init__()
        assert len(attns) == len(channels) - 1

        self.in_dim = in_dim
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        ch = channels[0]
        in_chs = [ch]

        self.in_block = nn.Conv2d(in_dim, channels[0], kernel_size=3, padding=1)

        for i, (ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            layer = [ResBlock(ch, 0.0, out_ch, out_ch)]
            if attns[i] > 0:
                layer.append(SpatialToSeq())
                layer.extend([SelfAttention(out_ch, out_ch, 64) for _ in range(attns[i])])
                layer.append(SeqToSpatial())
            layer.append(ResBlock(out_ch, 0.0, out_ch, out_ch, down=True))
            self.down_blocks.append(nn.Sequential(*layer))
            in_chs.append(out_ch)

        layer = [ResBlock(out_ch, 0.0, out_ch, out_ch)]
        if middle_attns > 0:
            layer.append(SpatialToSeq())
            layer.extend([SelfAttention(out_ch, out_ch, 64) for _ in range(middle_attns)])
            layer.append(SeqToSpatial())
        layer.append(ResBlock(out_ch, 0.0, out_ch, out_ch))
        self.middle_block = nn.Sequential(*layer)

        for i, (ch1, ch2) in enumerate(zip(channels[::-1][:-1], channels[::-1][1:])):
            i = len(attns) - 1 - i
            ch = ch1 + in_chs.pop()  # 1024, 512,
            out_ch = ch2  # 256, 128
            layer = [ResBlock(ch, 0.0, out_ch, out_ch)]
            if attns[i] > 0:
                layer.append(SpatialToSeq())
                layer.extend([SelfAttention(out_ch, out_ch, 64) for _ in range(attns[i])])
                layer.append(SeqToSpatial())
            layer.append(ResBlock(out_ch, 0.0, out_ch, out_ch, up=True))
            self.up_blocks.append(nn.Sequential(*layer))

        self.out_block = zero_module(nn.Conv2d(out_ch, in_dim, kernel_size=3, padding=1))

    def forward(self, x):
        res = []
        with nullcontext():  # Uncomment this line to break compile
        # with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):  # Uncomment this line to fix compile
            x = self.in_block(x)

            for layer in self.down_blocks:
                x = layer(x)
                res.append(x)

            x = self.middle_block(x)

            for layer in self.up_blocks:
                x = torch.cat([x, res.pop()], dim=1)
                x = layer(x)

            x = self.out_block(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--use_compile", action="store_true")
    args = parser.parse_args()
    return args


def main(rank, world_size, args):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    model = UNet(4, [128, 256, 512, 512], [0, 1, 1], 0).to(device)

    if args.use_compile:
        print("Trying compile.")
        model.compile(mode="default")

    model = FSDP(
        module=model,
        device_id=rank,
        use_orig_params=args.use_compile,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=ModuleWrapPolicy({nn.Sequential}),
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            buffer_dtype=dtype,
            reduce_dtype=dtype,
        ),
    )

    amp_context = torch.amp.autocast("cuda", dtype=dtype, enabled=True)
    scaler = ShardedGradScaler(enabled=dtype == torch.float16)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98))

    iterator = range(args.num_iterations)
    if rank == 0:
        iterator = tqdm(iterator, total=args.num_iterations)

    for _ in iterator:
        x = torch.randn(args.batch_size, 4, 64, 64, device=device)

        out = model(x)
        with amp_context:
            loss = F.mse_loss(x, out)

        loss_test = loss.clone()  # Ensure local loss is not changed by allreduce
        torch.distributed.all_reduce(loss_test)  # Check if any gpu has NaN loss
        if rank == 0:
            iterator.set_description(f"Loss: {loss_test.item()}")
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