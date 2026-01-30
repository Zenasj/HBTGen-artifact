import torch

model = fully_shard(model)

# Set `reshard_after_forward` to False
with torch.no_grad():
    for _ in range(5):
        x = model(x)
# Set `reshard_after_forward` to True

model(x)