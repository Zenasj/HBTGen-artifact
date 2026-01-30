import torch

x = torch.randn(10000000000, device='cuda')
# RuntimeError: CUDA out of memory. Tried to allocate 37.25 GiB (GPU 0; 23.68 GiB total capacity; 192.29 MiB already allocated; 15.77 GiB free; 5.05 GiB reserved in total by PyTorch)

torch.cuda.set_per_process_memory_fraction(0.1)
x = torch.randn(10000000000, device='cuda')
# RuntimeError: CUDA out of memory. Tried to allocate 37.25 GiB (GPU 0; 23.68 GiB total capacity; 192.29 MiB already allocated; 15.79 GiB free; 2.37 GiB allowed; 5.05 GiB reserved in total by PyTorch)

if TEST_CUDA and 'NUM_PARALLEL_PROCS' in os.environ:
    num_procs = int(os.getenv("NUM_PARALLEL_PROCS", "3"))
    # other libraries take up about 11% of space per process
    torch.cuda.set_per_process_memory_fraction(round(1 / num_procs - .11, 2))