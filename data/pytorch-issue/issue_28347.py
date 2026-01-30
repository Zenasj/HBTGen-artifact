import torch
  
for N in range(45,100):
    print(f"Trying {N}...")
    line = torch.zeros(size=(1,N))
    print(f"  Before arange shape is {line.shape}")
    assert len(line.shape) == 2
    torch.arange(-1, 1, 2/N, dtype=torch.float32, out=line)
    print(f"  After arange shape is {line.shape}")
    assert len(line.shape) == 2