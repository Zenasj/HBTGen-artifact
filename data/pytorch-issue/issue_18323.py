import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
    def forward(self, input):
        x_in = input.sub(127.) ##after uncomment this line, the convert process works.
        return self.pad(x_in)

if __name__ == '__main__':
    generator = Network()
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input_1"]
    output_names = ["output_1"]
    torch.onnx.export(generator, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)