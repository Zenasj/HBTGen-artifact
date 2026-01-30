import torch
import torch.nn as nn


class Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 64) 
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x): 
        x = self.emb(x)
        return self.fc1(x)


def main():
    model = Mod()
    x = torch.randint(0, 10, (1, 1,))
    traced_module = torch.jit.trace(model, x)
    traced_module.save('traced_model.pt')

main()