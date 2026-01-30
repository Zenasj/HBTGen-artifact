import torch.nn as nn

import torch
import onnx

class Model(torch.nn.Module):
    def forward(self, x):
        return x.cumsum(0, dtype=None), x.cumsum(0, dtype=torch.int32), x.cumsum(0, dtype=torch.float)

input = torch.rand(1, 2, 3).to(torch.int32)
model = Model()
output = model(input)

torch.onnx.export(model, input, 'error.onnx', opset_version=11)

omodel = onnx.load('error.onnx')
lookup = {
    onnx.TensorProto.BOOL:      'onnx.TensorProto.BOOL',
    onnx.TensorProto.DOUBLE:    'onnx.TensorProto.DOUBLE',
    onnx.TensorProto.FLOAT16:   'onnx.TensorProto.FLOAT16',
    onnx.TensorProto.FLOAT:     'onnx.TensorProto.FLOAT',
    onnx.TensorProto.INT8:      'onnx.TensorProto.INT8',
    onnx.TensorProto.INT16:     'onnx.TensorProto.INT16',
    onnx.TensorProto.INT32:     'onnx.TensorProto.INT32',
    onnx.TensorProto.INT64:     'onnx.TensorProto.INT64',
    onnx.TensorProto.UINT8:     'onnx.TensorProto.UINT8',
    onnx.TensorProto.UINT16:    'onnx.TensorProto.UINT16',
    onnx.TensorProto.UINT32:    'onnx.TensorProto.UINT32',
    onnx.TensorProto.UINT64:    'onnx.TensorProto.UINT64'
}

print('PyTorch Output DTypes: {}'.format(tuple(o.dtype for o in output)))
print('ONNX Output DTypes: {}'.format(
    tuple(lookup.get(o.type.tensor_type.elem_type) for o in omodel.graph.output))
)