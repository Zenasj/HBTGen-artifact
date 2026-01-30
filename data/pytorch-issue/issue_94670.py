import torch

# OK
data = {"a": 1, "b": "str"}
torch.save(data, "data.pth")
torch.load("data.pth", weights_only=True)
print("PASS 1")

# Error
data = {"a": 1, "b": "str", "c": 1e-6}
torch.save(data, "data.pth")
torch.load("data.pth", weights_only=True)
print("PASS 2")