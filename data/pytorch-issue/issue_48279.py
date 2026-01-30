####### vector_add_stream.py##########
import torch

# Allocate input tensors
N = 3000000000  # 3 billion elements (# H100 94 GB)
A1 = torch.randn(N, device="cuda")
B1 = torch.randn(N, device="cuda")
C1 = torch.empty_like(A1)

A2 = torch.randn(N, device="cuda")
B2 = torch.randn(N, device="cuda")
C2 = torch.empty_like(A2)

# Create two CUDA streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# Launch vector addition in stream1
with torch.cuda.stream(stream1):
    C1.copy_(A1 + B1)  # PyTorch kernel for element-wise addition

# Launch vector addition in stream2
with torch.cuda.stream(stream2):
    C2.copy_(A2 + B2)

# Wait for both kernels to finish
torch.cuda.synchronize()

print("Both vector additions ran on separate streams!")