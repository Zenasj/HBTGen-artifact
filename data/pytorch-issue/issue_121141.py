import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import os
from src.utils import get_logger, get_torch_profiler, schedule

os.environ["TORCH_COMPILE_DEBUG"] = "1"
batch_size = [1, 4, 8, 16, 32, 64]
model = models.resnet18()
model = model.cuda()

logger = get_logger()


def run(batch_size: int):
    inputs = (torch.rand((batch_size, 3, 224, 224)) * 255).to(dtype=torch.uint8)
    with profile(schedule=schedule(wait=0, warmup=8, repeat=4, active=4)) as prof:
        with record_function("to gpu"):
            inputs = inputs.cuda()
            inputs = inputs.to(torch.float32)
        with record_function("to float"):
            inputs = inputs.to(torch.float32)
        with record_function("forward"):
            model(inputs)

    prof.export_chrome_trace("trace.json")
    keys = prof.key_averages()
    print(prof.key_averages().table(row_limit=10))
    my_keys = list(filter(lambda e: e.key in ["to gpu", "to float", "forward"], keys))
    print(prof.key_averages().table(row_limit=10, top_level_events_only=True))


run(1)

my_schedule = schedule(
    skip_first=0,
    wait=1,
    warmup=4,
    active=5)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print("trace_handler")
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=my_schedule) as prof:
    for _ in range(10):
        with record_function("model_inference"):
            model(inputs)
        prof.step()