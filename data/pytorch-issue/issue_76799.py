import argparse

import torch

@torch.jit.script  # JIT decorator
def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def unfused_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def more_memory(gpu: bool) -> None:
    x = make_data(gpu)
    result = unfused_gelu(x)
    g = result.sum().backward()


def less_memory(gpu: bool, alpha=1) -> None:
    x = make_data(gpu)
    result = fused_gelu(x)
    g = result.sum().backward()


def make_data(gpu: bool = False) -> torch.Tensor:
    n1 = int(1e7)
    if gpu:
        x = torch.randn(n1, requires_grad=True, device=torch.device("cuda"))  # column vector
    else:
        x = torch.randn(n1, requires_grad=True)

    return x

def main(args: argparse.Namespace) -> None:

    more_memory(args.gpu)
    less_memory(args.gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()

    main(args)

import functools
import time

import torch

printnow = functools.partial(print, flush=True)

class TorchGpuProfiler:
    def __init__(self, on=True, msg=None) -> None:
        self.on = on
        self.msg = msg

    def __enter__(self):
        if self.on:
            self._start = time.perf_counter()
            self._mem_start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.on:
            self.elapsed_time = time.perf_counter() - self._start
            self.max_memory = torch.cuda.max_memory_allocated()
            printnow(
                self.msg, f"{int(1e3 * self.elapsed_time)} ms", f"{self.max_memory / (1<<20): .2f} MiB", sep="    "
            )