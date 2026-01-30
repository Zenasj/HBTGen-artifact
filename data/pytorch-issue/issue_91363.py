import torch

weights = self.model.state_dict()
for name, param in weights.items():
  weights[name] = param.to(torch.float32)
self.model = self.model.load_state_dict(weights)