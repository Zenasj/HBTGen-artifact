import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule

my_schedule = schedule(
    skip_first=1,
    wait=1,
    warmup=1,
    active=20,
    repeat=1)

dist.init_process_group("nccl")
n_gpus = torch.cuda.device_count()
print(f"Running on {n_gpus} gpus")
W = 2048
H = 8192
activation_tensor = torch.rand(W, H, 128, dtype=torch.bfloat16, device=f'cuda:{dist.get_rank()}') # large tensor activation
weight_tensor = torch.rand(W, H, 128//n_gpus, dtype=torch.bfloat16, device=f'cuda:{dist.get_rank()}') # large tensor
cpu_buffer = torch.empty(W, H, 128, dtype=torch.bfloat16, device='cpu', pin_memory=True)
to_tensor = torch.zeros_like(activation_tensor)
N_times = 30
comm_stream = torch.cuda.Stream()
offload_stream = torch.cuda.Stream()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,schedule=my_schedule) as prof:
    for op_idx in range(N_times):
        with torch.cuda.stream(comm_stream):
            torch.distributed.all_gather_into_tensor(to_tensor, weight_tensor) # AG
        with torch.cuda.stream(offload_stream):
            cpu_buffer.copy_(activation_tensor, non_blocking=True) # DtoH
        prof.step()
    torch.cuda.synchronize() # wait all kernels in all streams
prof.export_chrome_trace("my_profile.json")
dist.destroy_process_group()