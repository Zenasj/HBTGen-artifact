import torch.nn as nn
import torch.nn.functional as F

py3
import torch
import onnxruntime as ort

x = torch.linspace(-100, 20, 201, dtype=torch.float32)
torch.onnx.export(
    torch.nn.RReLU().eval(),
    (x,),
    'rrelu.onnx',
    input_names=['x'],
    output_names=['out'],
    dynamic_axes={
        'x': {0: 'T'},
        'out': {0: 'T'},
    },
    opset_version=16,
)


sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_sess = ort.InferenceSession('rrelu.onnx', sess_options=sess_options)
out1 = ort_sess.run(['out'], {'x': x.numpy()})[0]
out2 = ort_sess.run(['out'], {'x': x.numpy()})[0]


import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
plt.plot(x, out1, label='onnx 1')
plt.plot(x, out2, label='onnx 2')
plt.plot(x, torch.rrelu(x, training=False), color='red', label='torch')
plt.title('RReLU')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

py3
class RReLU(nn.RReLU):
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return torch.rrelu(input, self.lower, self.upper, self.training, self.inplace)
        return F.leaky_relu(input, (self.lower + self.upper) / 2, self.inplace)