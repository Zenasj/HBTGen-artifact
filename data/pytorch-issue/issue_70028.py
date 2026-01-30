from train_shape_corr import create_model
from torch.profiler import profile, record_function, ProfilerActivity
import torch
from pathlib import Path
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("output/tmp/trace_" + str(p.step_num) + ".json")

def evaluate_test_runtime():
    model = main(RUN=False)
    model.cuda()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler
    ) as p:
        for path in Path('data/cocked').rglob('*'):
            data = torch.load(str(path))
            model.inner_forward(data)
            p.step()

evaluate_test_runtime()