import torch.nn as nn

import torch
from torch import nn

class BugReproduce(nn.Module):
    def __init__(self) -> None:
        super(BugReproduce, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x1, x2):
        x1_fc1 = self.fc1(x1)
        x2_fc1 = self.fc1(x2)
        print(x1_fc1[:2, :] == x2_fc1)

        x1_fc2 = self.fc2(x1_fc1)
        x2_fc2 = self.fc2(x2_fc1)
        print(x1_fc2[:2, :] == x2_fc2)

        return x1_fc2, x2_fc2


if __name__ == "__main__":
    x1 = torch.randn(4, 32)
    x2 = torch.empty(2, 32)
    x2.copy_(x1[:2, :])
    model = BugReproduce()
    y1, y2 = model(x1, x2)