import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def forward(self, x):
        xt = x.transpose(1,2)
        return xt

m  = Model().cuda()
i = (torch.randn(1,2,3).cuda(),)

torch.onnx.export(m, i, "model.onnx",
                  input_names=["INPUT_0"],
                  output_names=["OUTPUT_0"],
                  dynamic_axes={"INPUT_0": {0: "batch_size"},
                                "OUTPUT_0": {0: "batch_size"}})