import torch.nn as nn

import time
import torch
from torch.utils.flop_counter import FlopCounterMode


if __name__ == "__main__":
    x = torch.rand(6, 192, 30, 96, device="cuda")
    model = torch.nn.ConvTranspose2d(192, 384, (2, 2), stride=2).to("cuda")

    with FlopCounterMode(model):
        start = time.time()
        for i in range(500):
            out = model(x)
            out.sum().backward()

        torch.cuda.synchronize()
        print(f"Elapsed: {time.time() - start}")