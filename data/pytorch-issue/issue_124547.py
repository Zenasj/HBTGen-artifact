import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch import Tensor

def my_normalize(input: Tensor, mean: Tensor, std: Tensor):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return (input - mean) / std

image = torch.randn(1, 3, 224, 224)
mean = torch.tensor([123.675, 116.28, 103.53])
std = torch.tensor([58.395, 57.12, 57.375])

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1,  warmup=9, active=90, repeat=1),
    record_shapes=True,
) as prof:
    for i in range(100):
        my_normalize(image, mean, std)
        prof.step() 

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

import torch
from torch.profiler import ProfilerActivity, schedule
from torch import Tensor

def my_normalize(input: Tensor, mean: Tensor, std: Tensor):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return (input - mean) / std

device = torch.device("cuda")

image_cuda = image.to(device,)
mean_cuda = mean.to(device)
std_cuda = std.to(device)

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1,  warmup=9, active=90, repeat=1),
    record_shapes=True,
) as prof:    
    for i in range(1000):
        r = my_normalize(image_cuda, mean_cuda, std_cuda)
        prof.step() 

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

import torch
from torch.profiler import ProfilerActivity, schedule
from torch import Tensor

def my_normalize(input: Tensor, mean: Tensor, std: Tensor):
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return (input - mean) / std

device = torch.device("cuda")

image_cuda = image.to(device,)
mean_cuda = mean.to(device)
std_cuda = std.to(device)

with profile(with_stack=True, 
             profile_memory=True, 
             experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True), 
             schedule=schedule(wait=1,  warmup=9, active=90, repeat=1)) as prof:   
    for i in range(1000):
        r = my_normalize(image_cuda, mean_cuda, std_cuda)
        prof.step() 

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))