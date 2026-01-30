import torchvision

import torch
from torchvision.models import resnet18

def print_memory_usage():
    for d in [0, 1]:
        stats = torch.cuda.memory_stats(device=d)
        m = stats["allocated_bytes.all.allocated"] + stats["inactive_split_bytes.all.allocated"] + stats["reserved_bytes.all.allocated"]
        print(f"\t- CUDA Device: {d}, allocated + reserved + non-released in MB: {m / 1024 / 1024}")

device = "cuda:1"
model = resnet18()
compiled_model = torch.compile(model)

print("--- Before compiled model to device")
print_memory_usage()

compiled_model.to(device)
x = torch.rand(16, 3, 320, 320, device=device)

print("--- Before compiled model forward")
print_memory_usage()

y = compiled_model(x)

print("--- Before compiled model backward")
print_memory_usage()

y.sum().backward()

print("--- After compiled model backward")
print_memory_usage()