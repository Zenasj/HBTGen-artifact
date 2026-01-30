import torch

errors = torch.tensor([-0.1944, -0.1944, -0.1945, -0.1945, -0.1945])

best_indices = torch.topk(errors, k=1, largest=True).indices.tolist()

print("Selected index:", best_indices)