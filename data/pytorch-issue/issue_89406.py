import torch.nn as nn

import torch

class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(100, 100)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(100, 100)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp1 = Mlp()
        self.mlp2 = Mlp()

    def forward(self, x):
        return self.mlp2(self.mlp1(x))

from torch.utils.data import DataLoader
import torchdata.datapipes.iter as dp
from torch.profiler import profile, ProfilerActivity

def make_mock_dataloader():
    pipe = dp.IterableWrapper([torch.rand(100) for _ in range(1000)])
    return DataLoader(pipe, batch_size=32, num_workers=2, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device=device)

target_traces = 2
traces_saved = 0

def trace_handler(prof: "torch.profiler.profile"):
    global traces_saved
    from os.path import join

    print("SAVING TRACE")

    tb_dir = join("./output2", "traces", str(traces_saved))
    handler = torch.profiler.tensorboard_trace_handler(
        tb_dir, worker_name=f"rank0"
    )
    handler(prof)

    prof.export_stacks(path=join(tb_dir, f"rank0.cuda.stacks"), metric="self_cuda_time_total")
    prof.export_stacks(path=join(tb_dir, f"rank0.cpu.stacks"), metric="self_cpu_time_total")

    # print(prof.events())

    traces_saved += 1
    if traces_saved == target_traces:
        prof.stop()

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # profile_memory=True,
    with_stack=True,
    # with_modules=True,
    # record_shapes=True,
    schedule=torch.profiler.schedule(
        skip_first=5, wait=1, warmup=5, active=5, repeat=target_traces
    ),
    on_trace_ready=trace_handler,
)
prof.start()

for idx, batch in enumerate(make_mock_dataloader()):
    print(f"idx: {idx}")
    batch = batch.to(device=device)
    out = net(batch)
    out.sum().backward()
    net.zero_grad()
    prof.step()