import torch.nn as nn

import torch 
import onnx

class testSubMod(torch.nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.lin = torch.nn.Linear(32, 32, bias=True) # < This, this bit here!

    def forward(self, out):
        for _ in torch.arange(8):
            out = self.lin(out)
        return out


class test(torch.nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.submod = torch.jit.script(testSubMod())

    def forward(self, x):
        out = torch.ones(
            [
                x.size(0),
                x.size(1)
            ],
            dtype=x.dtype,
            device=x.device
        )

        return self.submod(out)

if __name__=='__main__':
    model = test()
    model = torch.jit.script(model)

    input_data = torch.ones((32, 32, 32)).float()
    output = model(input_data)

    torch.onnx.export(model, input_data,
                      'test.onnx', example_outputs=output)


    onnx_model = onnx.load("test.onnx")
    print(onnx.helper.printable_graph(onnx_model.graph))