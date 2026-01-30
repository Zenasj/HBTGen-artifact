import torch
import torch.nn as nn

def print_cuda_memory(name = "none"):
    print(f"[{name}] alloc: {torch.cuda.memory_allocated() / 1024**2}, reserved: {torch.cuda.memory_reserved() / 1024**2}, max reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2}")

def create_and_compile():
    graph = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1)).cuda()
    print_cuda_memory("before compile ")
    graph = torch.compile(graph, mode="reduce-overhead")
    graph(torch.randn((3, 3, 512, 512)).cuda())
    print_cuda_memory("after compile ")

for i in range(5):
    print(f"test {i}")
    create_and_compile()
    print_cuda_memory("after delete")
    sleep(1)