import torch

print("Example of `end` is seemingly included from rounding error.")
print(f"{torch.arange(0, 1, 1/49)=}")
print("Error is not fixed by adding epsilon.")
print(f"{torch.arange(0, 1.00001, 1/49)=}")
print("Error is fixed by subtracting epsilon.")
print(f"{torch.arange(0, 0.99999, 1/49)=}")
print("Example of `end` excluded as expected.")
print(f"{torch.arange(0, 1, 1/16)=}")
print("After adding epsilon, now 1 is included, which is not consistent.")
print(f"{torch.arange(0, 1.00001, 1/16)=}")
print("Subtracting epsilon has no effect (expected).")
print(f"{torch.arange(0, 0.99999, 1/16)=}")