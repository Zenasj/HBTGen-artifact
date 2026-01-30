import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet50().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, with_modules=False) as prof:
    model(inputs)
print(prof.key_averages(group_by_stack_n=2).table(sort_by="self_cuda_time_total"))