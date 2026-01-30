import torch
import torch.nn as nn

class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        self.rnn1 = nn.LSTM(8, 8, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(8 * 2, 8)

        self.rnn2 = nn.LSTM(8, 8, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(8 * 2, 8)

    def forward(self, input):
        rnn_output1, _ = self.rnn1(input)
        linear_output1 = self.linear1(rnn_output1)

        rnn_output2, _ = self.rnn2(linear_output1)
        linear_output2 = self.linear2(rnn_output2)

        return linear_output2

net = testModel()
net.eval()
input = torch.zeros((1, 100, 8), dtype=torch.float32)

with torch.no_grad():
   output = net(input)


torch.onnx.export(
        net,
        input,
        'out.onnx',
        export_params=True,
        opset_version=13, 
        verbose=False,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={
                        'input' : {0 : 'batch_size',1:'w',2:'h'},
                        'output' : {0 : 'batch_size', 1:'w',2:'h'},
                }
        )

##### ONNXRuntime ####
import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession('out.onnx')
result_1 = sess.run([], {'input': input.numpy()})[0] # Failed