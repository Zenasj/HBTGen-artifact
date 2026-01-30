import gc
import torch

func = torch._foreach_exp

gc.collect()
torch.cuda.empty_cache()
torch._C._cuda_clearCublasWorkspaces()

primals = [torch.randn(3, device="cuda", requires_grad=True)]

init_cached_memory = torch.cuda.memory_allocated()
bytes_free, bytes_total = torch.cuda.mem_get_info(0)
init_driver_mem_allocated = bytes_total - bytes_free
print(f"# {init_cached_memory = }, {init_driver_mem_allocated = }")


def call(primals):
    func(primals)


call(primals)
torch._C._cuda_clearCublasWorkspaces()
caching_allocator_mem_allocated = torch.cuda.memory_allocated()
bytes_free, bytes_total = torch.cuda.mem_get_info(0)
driver_mem_allocated = bytes_total - bytes_free
print(f"# {caching_allocator_mem_allocated = }, {driver_mem_allocated = }")