import torch.nn as nn

3
from nemo.collections.nlp.models.machine_translation import MTEncDecModel

model = MTEncDecModel.from_pretrained('nmt_en_de_transformer12x2')
model.decoder.export('pretrained_decoder.onnx')

import torch
import io

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_1d = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        w_2d = self.w_1d.unsqueeze(0)  # constant fold
        return w_2d + x

f = io.BytesIO()
x = torch.randn(2, 2)
torch.onnx.export(Model(), x, f, verbose=True)
torch.onnx.export(Model(), x, f, verbose=True, do_constant_folding=False)