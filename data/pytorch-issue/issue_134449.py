import torch

self.decode_one_token = torch.compile(
      self.decode_one_token, mode="max-autotune", fullgraph=True
)