import torch.nn as nn

import torch


class Model(torch.nn.Module):
    def __init__(self, num_classes, num_channels):
        super().__init__()
        self.embed = torch.nn.Embedding(num_classes, num_channels * num_channels)
        self.num_channels = num_channels

    def forward(self, x, classes):
        x.requires_grad_()
        weights = self.embed(classes).view(-1, self.num_channels, self.num_channels)
        aux = torch.bmm(weights, x.unsqueeze(-1)).square().sum()
        grad = torch.autograd.grad(aux, [x])[0]
        return grad


device = "cpu"  # "cuda"
num_batch = 512
num_channels = 256
num_classes = 3

x = torch.randn(num_batch, num_channels, dtype=torch.float32, device=device)
classes = torch.randint(0, num_classes, (num_batch,), dtype=torch.int64, device=device)

model = Model(num_classes, num_channels).to(device=device)
eager_out = model(x, classes)
print(eager_out)

batch_dim = torch.export.Dim("batch", min=1, max=1024)
exported = torch.export.export(
    model,
    (
        x,
        classes,
    ),
    strict=False,
    dynamic_shapes={"x": {0: batch_dim}, "classes": {0: batch_dim}},
)

model = torch.compile(exported.module())
out = model(x, classes)
loss = out.square().mean()
loss.backward()

print(model.embed.weight.grad)