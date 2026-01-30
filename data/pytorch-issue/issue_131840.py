import torchvision

import torch

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.xpu.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).xpu(),
        torch.randint(1000, (b,)).xpu(),
    )

N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).xpu()

model = init_model()

# Reset since we are using a different mode.
import torch._dynamo
torch._dynamo.reset()

model_opt = torch.compile(model, mode="reduce-overhead")

inp = generate_data(16)[0]
with torch.no_grad():
    print("eager:", timed(lambda: model(inp))[1])
    print("compile:", timed(lambda: model_opt(inp))[1])

def timed(fn):
    start =  time.time_ns()
    result = fn()
    torch.xpu.synchronize()
    return result, (time.time_ns() - start) / 1000000000