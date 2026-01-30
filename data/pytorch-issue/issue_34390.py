import torch.nn as nn

a = torch.empty((0,)).view(1, -1)
print(a.shape)

def patch_view_opset9():
    """Patch view to never use Flatten
    Issue: https://github.com/pytorch/pytorch/issues/34390
    """
    import torch.onnx.symbolic_helper as sym_help
    import torch.onnx.symbolic_opset9

    # noinspection PyProtectedMember
    def view(g, self, size):
        size = sym_help._maybe_get_const(size, 'is')
        if sym_help._is_value(size):
            shape = size
        else:
            shape = g.op("Constant", value_t=torch.LongTensor(size))
        return g.op("Reshape", self, shape)

    torch.onnx.symbolic_opset9.view = view

import torch
import onnxruntime as rt

class Module(torch.nn.Module):
    def forward(self, x, ):
        x = x.view(-1, 2)
        x = x.view(1, -1)
        return x

model = Module()
x = torch.empty((0,))
print(model(x).shape)

torch.onnx.export(model, torch.ones(2), '/mnt/output/gr/test.onnx',
                  verbose=True,
                  input_names=['data'],
                  output_names=['output'],
                  opset_version=11,
                  dynamic_axes={'data': {0: 'l'}})

sess=rt.InferenceSession("/mnt/output/gr/test.onnx")
outputs = sess.run(['output'], {
    'data': x.numpy(),
})
print(outputs[0].shape)