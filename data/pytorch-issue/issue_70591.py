import torch

for obj in (True, 1, 1.0):
    print(f"{obj} -> {torch.asarray(obj).dtype}")