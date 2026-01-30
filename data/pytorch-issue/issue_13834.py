import torch.nn as nn

import torch

class Split(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op('Split', input, outputs=2)

    @staticmethod
    def forward(ctx, input):
        return input[0], input[1]

def test_onnx_export():
    class MyModule(torch.nn.Module):
        def forward(self, input):
            return Split().apply(input)

    model_string = torch.onnx.export_to_pretty_string(
                   MyModule(),
                   (torch.tensor([0, 1])),
                   "/tmp/custom_op.onnx")
    print(model_string)

test_onnx_export()