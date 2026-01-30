import torch

torch.load(f, weights_only=True)
# Fails with Unsupported GLOBAL: {bla} ... and says to use `add_safe_globals` if it is safe
torch.add_safe_globals([bla])