import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(1000, 512)
        self.lin = torch.nn.Linear(512, 1024)
        self.loss = torch.nn.CTCLoss(0, reduction='sum')

    def make_pad_mask(self, ylen, y):
        r = torch.arange(0, y.size(1), dtype=ylen.dtype).unsqueeze(0)
        u = r.expand(ylen.size(0), r.size(1))
        return u >= ylen.unsqueeze(-1)

    def forward(self, x, xlen, y, ylen):
        # with torch.no_grad():
        m = self.make_pad_mask(ylen, y)
        y.masked_fill_(m, 0)
        c = y.new_full((y.size(0), 1), 0)
        z = torch.cat([c, y], dim=1)

        # print('uncommenting this print fixes the issue')
        z = self.embed(z)
        z = self.lin(z)

        logits = x + z
        logits = logits.transpose(0, 1)

        return self.loss(logits, y, xlen, ylen)

model = Model()
scripted = torch.jit.script(model)
torch.jit.save(scripted, 'export.pt')

x = torch.rand(1, 15, 1024)
xlen = torch.tensor([15], dtype=torch.int)
y = torch.tensor([[33, 50, 49, 42, 326, 32, 172, 205, 242, 66, 162, 151, 51, 504]], dtype=torch.int)
ylen = torch.tensor([y.size(1)], dtype=torch.int)

model = torch.jit.load('export.pt') # commenting this line out fixes the issue (ie. using the PyTorch model instead of the TorschScript one)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for i in range(5):
    loss = model(x, xlen, y, ylen)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()