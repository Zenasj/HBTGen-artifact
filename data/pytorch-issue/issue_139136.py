import torch.nn as nn

class LinearAndCEL(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(C, V)
                self.ce = nn.CrossEntropyLoss()

            def forward(self, x, y):
                return self.ce(self.linear(x).view(B * T, V), y.view(-1))