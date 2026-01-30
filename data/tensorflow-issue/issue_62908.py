py
import torch
import torch.nn as nn
import torch.nn.functional as F
import ai_edge_torch


class FCResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(105, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.dense1(x)
        y = F.dropout(F.relu(self.dense2(x)), p=0.2)
        y = F.dropout(F.relu(self.dense3(y)), p=0.2)
        x = x + y
        x = F.dropout(F.relu(self.dense4(x)), p=0.2)
        return self.dense5(x)


model = FCResidualBlock()
sample_inputs = (torch.randn(1, 105),)

edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export("fc_res_block.tflite")