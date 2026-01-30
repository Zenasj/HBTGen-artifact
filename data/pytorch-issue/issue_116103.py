import torch
from pathlib import Path

print(torch.__version__)

file = Path("example.pt")
torch.save(torch.rand(5, 3), file)

print(torch.load(file, mmap=False))  # works!
print(torch.load(file, mmap=True))   # does not work!