import torch
import torch.nn as nn

class SomeModel(nn.Module):

    def __init__(self):
        super(SomeModel, self).__init__()
        dim = 5
        n = 4 * 10 ** 6
        self.emb = nn.Embedding(n, dim)
        self.lin1 = nn.Linear(dim, 1)
        self.seq = nn.Sequential(
            self.emb,
            self.lin1,
        )

    def forward(self, input):
        return self.seq(input)

model = SomeModel()

dummy_input = torch.tensor([2], dtype=torch.long)
dummy_input

torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)

n = int(1.1 * 10 ** 8)