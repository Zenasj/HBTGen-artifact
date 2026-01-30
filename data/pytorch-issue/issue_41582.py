import torch.nn as nn

import torch
from torch import nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        return nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)

def main():
    N, T, F = 5, 30, 50
    lengths = torch.tensor([5, 4, 2, 2, 1], dtype=torch.int32).cuda()
    x = torch.ones((N, T, F)).cuda()

    model = DummyModel()
    scripted_model = torch.jit.script(model)

    # This ok
    scripted_model(x, lengths.cpu())

    # This will report an error
    scripted_model(x, lengths)

if __name__ == "__main__":
    main()