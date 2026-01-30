import torch

cupy_allocator_so = 'cupy/cuda/memory.cpython-39-x86_64-linux-gnu.so'
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(cupy_allocator_so, 'cupy_malloc_ext', 'cupy_free_ext')

torch.cuda.memory.change_current_allocator(new_alloc)