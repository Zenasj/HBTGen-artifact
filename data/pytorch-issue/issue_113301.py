import os
import tempfile

import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
import torch._inductor.config
from torch._inductor.codecache import AsyncCompile


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, input, grid):
        input = self.conv(input)
        # grid_sample results in generated triton code with device_assert, which causes the issue
        sampled = grid_sample(input, grid, align_corners=False)
        return sampled.sum()


if __name__ == "__main__":
    # Make sure we have a fresh cache dir
    cache_dir = tempfile.mkdtemp()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")

    # Launch workers with spawn. Make sure to set TORCHINDUCTOR_COMPILE_THREADS=1 before
    # running the script so inductor workers aren't already started
    torch._inductor.config.worker_start_method = "spawn"
    torch._inductor.config.compile_threads = 2
    AsyncCompile.warm_pool()

    input = torch.rand((4, 3, 12, 12)).to(device="cuda")
    grid = (torch.rand((4, 3, 3, 2)) - 0.5).to(device="cuda")
    model = Model().to("cuda")
    model = torch.compile(model, mode="default")
    loss = model(input, grid)
    loss.backward()