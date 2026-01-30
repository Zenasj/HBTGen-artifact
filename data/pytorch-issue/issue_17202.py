import torch.nn as nn

import torch
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

class TestNet(torch.nn.Module):
    def forward(self, x):
        y = x[:,:,-1]
        return y

def main():
    input = torch.rand(1,128,1,15,20)
    net = TestNet()

    output = net(input)

    torch_out = torch.onnx._export(net, input, "net.onnx", export_params=True)

    model_onnx = onnx.load("net.onnx")
    prepared_backend = onnx_caffe2_backend.prepare(model_onnx)
    W = {model_onnx.graph.input[0].name: input.data.numpy()}

    # Run the Caffe2 net:
    c2_out = prepared_backend.run(W)

if __name__ == '__main__':
    main()