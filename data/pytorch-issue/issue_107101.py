import numpy as np
import torch
import time
import ctypes
import ctypes.util

cuda_lib = ctypes.CDLL("cudart64_110.dll")
# CUDA flags
cudaHostRegisterDefault = 0x00
cudaSuccess = 0
cudaHostAlloc = cuda_lib.cudaHostAlloc
cudaHostAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
cudaHostAlloc.restype = ctypes.c_int
mem_ptr = ctypes.c_void_p()
# Register the allocated memory to CUDA
result = cudaHostAlloc(ctypes.addressof(mem_ptr), 1024 * 1024 * 1024, cudaHostRegisterDefault)
assert result == 0
with torch.no_grad():
    model1 = torch.jit.load(r"C:\Users\kohill.WIN-5VEFDUPJLBO\Desktop\segment0727.pt")
    model1 = model1.to("cuda:0")
    model1.eval()
    for _ in range(4):
        t0 = time.monotonic()
        for i in range(400):
            model1Inputs = torch.zeros((1, 3, 896, 1088), dtype=torch.float32, device="cuda:0")
            _ = model1(model1Inputs)
            torch.cuda.synchronize("cuda:0")
        print("inference time: ", time.monotonic() - t0)

    model2 = torch.jit.load(r"C:\Users\kohill.WIN-5VEFDUPJLBO\Desktop\backbump0808_2.pt")
    model2 = model2.to("cuda:1")
    model2.eval()
    for _ in range(1):
        t0 = time.monotonic()
        model2Inputs = torch.zeros((1, 3, 896, 1088), dtype=torch.float32, device="cuda:1")
        _ = model2(model2Inputs)
        torch.cuda.synchronize("cuda:1")
        print("bump inference time: ", time.monotonic() - t0)

    for _ in range(4):
        t0 = time.monotonic()
        for i in range(400):
            model1Inputs = torch.zeros((1, 3, 896, 1088), dtype=torch.float32, device="cuda:0")
            _ = model1(model1Inputs)
            torch.cuda.synchronize("cuda:0")
        print("inference time: ", time.monotonic() - t0)