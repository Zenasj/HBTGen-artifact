import torch
import torch.nn as nn

class Model_FyJv5nvMj42clGYaljLtCDb8pkc5Kd6N(nn.Module):
    def __init__(self):
        super(Model_FyJv5nvMj42clGYaljLtCDb8pkc5Kd6N, self).__init__()
        self.conv1_mutated = torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=[1], stride=[1], padding=[1], dilation=[1], groups=1, bias=True)
        self.tail_flatten = torch.nn.Flatten()
        self.tail_fc = torch.nn.Linear(in_features=131200, out_features=10)

    def forward(self, x):
        x = self.conv1_mutated(x)
        tail_flatten_output = self.tail_flatten(x)
        tail_fc_output = self.tail_fc(tail_flatten_output)

        tail_fc_output = tail_fc_output
        return tail_fc_output