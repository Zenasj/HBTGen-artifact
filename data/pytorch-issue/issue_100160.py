import torch.nn as nn

import torch
from torch import nn
import onnx
import onnxruntime
import numpy as np



class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(32, 10, 6, padding='same', padding_mode = "circular")
        self.linear = nn.Linear(640, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


def main():
    model = CNN()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    input_shape = (1, 32, 8, 8)
    test_data = torch.rand(*input_shape).to(device)
    model.to(device)
    output = model(test_data)
    print("Pytorch output:", output[0].cpu().detach().numpy())

    dummy_input = torch.ones(input_shape).to(device)
    torch.onnx.export(
        model, dummy_input, "model.onnx",
        export_params=True,
        opset_version=17,
    )
    sess = onnxruntime.InferenceSession("model.onnx")
    input_name = sess.get_inputs()[0].name

    #prepare input
    training_input = test_data.cpu().numpy()

    # Run inference
    result = sess.run(None, {input_name: training_input})

    print("ONNX runtime output: ", result[0][0])
    

if __name__ == "__main__":
    main()