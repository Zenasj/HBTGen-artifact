import onnxruntime
import torch
import torch.nn as nn
import torch.onnx


class TestModel(nn.Module):
    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.size())  # Issue here
        # position_ids = position_ids.repeat(input_ids.size(0), 1)  # Workaround
        return position_ids
    
torch.onnx.export(
    TestModel(),
    torch.ones(1, 512),
    'tmp.onnx',
    input_names=['input_ids'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size'}
    },
    opset_version=14,
)

sess = onnxruntime.InferenceSession('tmp.onnx')
print(torch.__version__)
print(onnxruntime.__version__)
print(sess.run(None, {'input_ids': [[0, 1]]})[0].shape)
print(sess.run(None, {'input_ids': [[0, 1]] * 1000})[0].shape)

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

class Model0(nn.Module):
    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.size())
        # position_ids = position_ids.repeat(input_ids.size(0), 1)
        return position_ids

class Model1(nn.Module):
    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        # position_ids = position_ids.expand(input_ids.size())
        position_ids = position_ids.repeat(input_ids.size(0), 1)
        return position_ids
    
sample_input = torch.ones(32, 512, dtype=torch.long)

t0 = benchmark.Timer(
    stmt='model(sample_input)',
    setup='model=Model().eval().cuda()',
    globals={'sample_input': sample_input, 'Model': Model0}
)
t1 = benchmark.Timer(
    stmt='model(sample_input)',
    setup='model=Model().eval().cuda()',
    globals={'sample_input': sample_input, 'Model': Model1}
)
print(t0.timeit(100000))
print(t1.timeit(100000))