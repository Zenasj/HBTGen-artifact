import torch.nn as nn

import onnx
import os
# import simplify
import torch
from torch import nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(3, 4, 4), BidirectionalLSTM(4, 4, 4)
        )

    def forward(self, x):
        x2 = torch.squeeze(x, 2)
        x3 = torch.permute(x2, (0, 2, 1))
        out = self.SequenceModeling(x3)
        return out  # shape 1,6,4


model = Model()
model.to("cpu")
model.eval()

input = torch.randn(1, 3, 1, 6)
output = model(input)
print("output shape:", output.shape)

input_shapes = [(1, 3, 1, 6)]
onnx_export_path = "./Bi_Two_lstm_batch.onnx"
dummy_input = []
for ele in input_shapes:
    dummy_input.append(torch.randn(ele))
dummy_input = tuple(dummy_input)
dynamic_axes = {
    "input": {0: "batch", 3: "width"},
    "output": {0: "batch", 1: "timestep"},
}
# torch.onnx.export(model, dummy_input, onnx_export_path,export_params=True,verbose=False, opset_version=11,do_constant_folding=True, input_names=["input"], output_names=["output"], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

torch.onnx.export(
    model,
    dummy_input,
    onnx_export_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
    input_names=["input"],
    dynamic_axes=dynamic_axes,
    output_names=["output"],
    verbose=True
)
print("export onnx to:", onnx_export_path)

# onnx_model = onnx.load(onnx_export_path)
# model_sim, check = simplify(onnx_model)
# assert check, "simplified onnx model could not be validated"
# save_path = os.path.splitext(onnx_export_path)[0] + "_sim.onnx"
# onnx.save(model_sim, save_path)

inp = torch.randn(input_shapes[0]).cpu().numpy()
rtsess = ort.InferenceSession(
    onnx_export_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
opu = rtsess.run(None, {"input": inp})
print(opu)