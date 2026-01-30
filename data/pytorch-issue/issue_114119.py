import torchvision

import torch
from torchvision.models import resnet18
import sys
from contextlib import nullcontext

# enable inference mode when compile or benchmark?
enable_when_compile = sys.argv[1] == "true"
enable_when_benchmark = sys.argv[2] == "true"


def _benchmark(
    iters,
    f,
    context,   # with torch.inference or not
    *args,
    **kwargs,
) -> float:
    """Estimates the average time duration for a single inference call in second
    Returns:
        estimated average time duration in second for a single inference call
    """
    with context():
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with context():
        start_event.record()
        for _ in range(iters):
            f(*args, **kwargs)
        end_event.record()
    torch.cuda.synchronize()
    elapsed_time_s = start_event.elapsed_time(end_event) * 1.0e-3
    avg_time_s = elapsed_time_s / iters
    print("Estimated average time duration: {:.6f} s".format(avg_time_s))
    return avg_time_s


class BenchmarkRunner(object):
    def __init__(self, use_inference_mode: bool):
        self.context = nullcontext if not use_inference_mode else torch.inference_mode

    def __call__(self, iters, f, *args, **kwargs) -> float:
        return _benchmark(iters, f, self.context, *args, **kwargs)


@torch.no_grad()
def run():
    input = [torch.rand(8, 3, 224, 224).to(torch.device("cuda"), dtype=torch.float16)]
    net = resnet18(pretrained=False).cuda().half()
    net.eval()
    context = nullcontext if not enable_when_compile else torch.inference_mode
    compiled = torch.compile(net, mode="reduce-overhead", backend="inductor")
    with context():
        _ = compiled(*input)

    latency_compiled = BenchmarkRunner(enable_when_benchmark)(10, compiled, *input)
    latency_torch = BenchmarkRunner(enable_when_benchmark)(10, net, *input)

    print(f"enable inference mode when compile: {enable_when_compile}")
    print(f"enable inference mode when benchmark: {enable_when_benchmark}")
    print(f"not compiled latency: {latency_torch}")
    print(f"compiled latency: {latency_compiled}, speed up: {latency_torch / latency_compiled}")

if __name__ == "__main__":
    run()

[tasklist]
### Tasks