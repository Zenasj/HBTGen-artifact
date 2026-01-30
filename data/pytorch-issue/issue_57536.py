import torch.nn as nn

import hashlib
import torch


def get_sha256_hash(file: str, chunk_size: int = 4096) -> str:
    hasher = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


file = "tensor.pt"
hashes = []
for _ in range(5):
    obj = torch.ones(1)
    torch.save(obj, file)
    hashes.append(get_sha256_hash(file)[:8])
    del obj

hash = hashes[0]
assert all(other == hash for other in hashes[1:])
print(hash)

import torch
torch.manual_seed(0)
x = torch.tensor([8., 8., 5., 0.])
torch.save(x, "out_tensor.pt")

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
torch.save(model, "out_model.pt")

import torch
torch.manual_seed(0)
x = torch.rand(1_000, 1_000)
y = x.T
z = x.view(1_000_000, 1)

torch.save({"x": x}, "out_tensor_x.pt")
torch.save({"x": x, "y": y, "z": z}, "out_tensor_xyz.pt")