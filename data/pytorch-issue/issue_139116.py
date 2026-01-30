import torch.nn as nn

#!/usr/bin/env python
from __future__ import annotations

import argparse
import functools
import random
import statistics
import time

import torch

from torch.profiler import profile, record_function, ProfilerActivity

def main():

    parameters = {
        "batch_size": 7_200_000,
        "num_blocks": 1,
        "linear1_width": 16
    }

    test(parameters, "bfloat16", True)

def test(
    parameters,
    dtype: str,
    padded: bool,
    width: int = 12,
    n: int = 10,
) -> None:
    torch.manual_seed(17)

    dtype = getattr(torch, dtype)
    batch = parameters.pop("batch_size")
    model = BaseModel(in_width=width, out_width=1, **parameters).cuda()
    inputs = torch.empty((batch, width), device="cuda", dtype=dtype).normal_()

    autocast = functools.partial(
        torch.autocast,
        device_type="cuda",
        dtype=dtype,
        enabled=dtype in [torch.bfloat16, torch.float16],
    )
    with autocast():
        outputs = model(inputs)
    del outputs

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )

    times = []
    for _ in range(n):
        batch_inputs = inputs if padded else inputs[: -random.randrange(1, 101)]

        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            with profile(activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    fwd = model(inputs)
        print("Foward")
        with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_output"):
                outputs = fwd.mean().backward()
        print("Backward")
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.time() - start
        times.append(duration)

    return statistics.median(times)

class BaseModel(torch.nn.Module):
    def __init__(
        self,
        num_blocks: int,
        linear1_width: int,
        in_width: int,
        out_width: int,
        relu_leak: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.relu_leak = relu_leak
        self.block = Block(
                    in_width=in_width,
                    linear1_width=linear1_width,
                    relu_leak=self.relu_leak,
                )

        self.output = torch.nn.Linear(
            in_features=linear1_width,
            out_features=out_width,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.block(inputs))

class Block(torch.nn.Module):
    def __init__(
        self,
        in_width: int,
        linear1_width: int,
        relu_leak: float,
    ) -> None:
        super().__init__()
        self.relu_leak = relu_leak
        self.linear1 = torch.nn.Linear(
            in_features=in_width,
            out_features=linear1_width,
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leaky_relu = functools.partial(
            torch.nn.functional.leaky_relu, negative_slope=self.relu_leak
        )
        l1 = leaky_relu(self.linear1(inputs))
        output = leaky_relu(l1)
        return output

if __name__ == "__main__":
    main()