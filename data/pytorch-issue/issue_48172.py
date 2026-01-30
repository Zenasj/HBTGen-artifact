import torch
torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory
# less than 0.5 will be ok:
tmp_tensor = torch.empty(int(total_memory * 0.499), dtype=torch.int8, device='cuda')
del tmp_tensor
torch.cuda.empty_cache()
# this allocation will raise a OOM:
torch.empty(total_memory // 2, dtype=torch.int8, device='cuda')

"""
It raises an error as follows: 
RuntimeError: CUDA out of memory. Tried to allocate 5.59 GiB (GPU 0; 11.17 GiB total capacity; 0 bytes already allocated; 10.91 GiB free; 5.59 GiB allowed; 0 bytes reserved in total by PyTorch)
"""