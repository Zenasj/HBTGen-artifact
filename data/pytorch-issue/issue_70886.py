import torch.nn as nn

import torch
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        y = self.l1(x.view(x.size(0), -1))
        y = torch.nan_to_num(y)
        return y

filepath = "model.onnx"
model = SimpleModel()
input_sample = torch.randn((1, 64))
model.to_onnx(filepath, input_sample, export_params=True)