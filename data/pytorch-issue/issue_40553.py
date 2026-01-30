import torch

logits = torch.tensor([0, 0, -1e16])
d = Categorical(logits=logits)
d.entropy() # tensor(0.6931)