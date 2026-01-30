import torch.nn as nn

from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn


class BinaryVisionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(kernel_size=(28, 28)),
            nn.Conv2d(3, 1, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(in_features=64, out_features=2, bias=True)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(self.model(input)), self.classifier.weight.mean()


def setup():
    dist.init_process_group(backend="nccl")


def cleanup():
    dist.destroy_process_group()


def demo_basic():
    setup()

    rank = dist.get_rank() if dist.is_initialized() else 0

    model = BinaryVisionClassifier().to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()

    data = torch.randn((16, 3, 224, 224), device=torch.device(rank))
    output1, output2 = ddp_model(data)

    output1.sum().backward(retain_graph=True)

    # ddp_model.module.classifier.weight.mean().backward()  # OK
    output2.backward()  # fails with static_graph=True
    #   File "a.py", line 56, in <module>
    #     demo_basic()
    # File "a.py", line 50, in demo_basic
    #     output2.backward()
    # File "glass38/lib/python3.8/site-packages/torch/_tensor.py", line 487, in backward
    #     torch.autograd.backward(
    # File "glass38/lib/python3.8/site-packages/torch/autograd/__init__.py", line 200, in backward
    #     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    # SystemError: <built-in method run_backward of torch._C._EngineBase object at 0x7f02852569f0> returned NULL without setting an error
    # ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 1414273) of binary: glass38/bin/python

    cleanup()
    print("success")


if __name__ == "__main__":
    demo_basic()