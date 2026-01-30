import torch

# Init caching
# b = torch.zeros(10, device='cuda')
new_alloc = torch.cuda.memory.CUDAPluggableAllocator('alloc.so', 'my_malloc', 'my_free')
old = torch.cuda.memory.get_current_allocator()
torch.cuda.memory.change_current_allocator(new_alloc)
b = torch.zeros(10, device='cuda')
# This will error since the current allocator was already instantiated
torch.cuda.memory.change_current_allocator(old)