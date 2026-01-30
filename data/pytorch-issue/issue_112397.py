import torch.nn as nn

import torch

from torch.utils.checkpoint import checkpoint

class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.ln1 = torch.nn.LayerNorm(4)

        self.linear2 = torch.nn.Linear(4, 4)
        self.ln2 = torch.nn.LayerNorm(4)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ln1(x)

        x = self.linear2(x)
        x = self.ln2(x)
        return x

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(4, 4)
        self.layers = torch.nn.ModuleList([DummyLayer() for _ in range(2)])
        self.head = torch.nn.Linear(4, 4)

        self._gradient_checkpointing_func = checkpoint

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    
    def forward(self, x):
        x = self.emb(x)

        for layer in self.layers:
            x = self._gradient_checkpointing_func(self.custom(layer), x)

        return x

model = DummyModel().to(0)
model.train()

for p in model.parameters():
    p.requires_grad_(True)

assert model.training

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dummy_input = torch.LongTensor([[0, 1, 0, 1]]).to(0)

model.train()
logits = model(dummy_input)
loss = logits.sum()

loss.backward()
optimizer.step()

for n, param in model.named_parameters():
    if param.grad is None:
        raise ValueError(f"Parameter {n} has no gradient!")