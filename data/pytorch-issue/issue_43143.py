import torch
a = torch.rand(size=(2,2), dtype=torch.complex32)
# RuntimeError: [enforce fail at CPUAllocator.cpp:64] . DefaultCPUAllocator: can't allocate memory: you tried to allocate 34359738368 bytes. Error code 12 (Cannot allocate memory)