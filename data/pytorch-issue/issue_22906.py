import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

torch_model = TestModel()
dummy_input = torch.randn(1, 3, 256, 256)

torch_out = torch.onnx.export(torch_model, dummy_input, 'test_model.onnx', verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, x.size()[2:], mode='bilinear', align_corners=False)
        return x

torch_model = TestModel()
dummy_input = torch.randn(1, 3, 256, 256)

torch_out = torch.onnx.export(torch_model, dummy_input, 'test_model.onnx', verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime

class TestModel(nn.Module):
    def __init__(self, align=False):
        super(TestModel, self).__init__()
        self.align=align
    def forward(self, x):
        x = F.interpolate(x, (4,4), mode='bilinear', align_corners=self.align)
        return x

x = torch.tensor([[0.,1.],[2.,3.]]).view([1,1,2,2])
model = TestModel(align=True)
out = model(x)

print("matrix to be interpolated :")
print(x)
print()
print( "pytorch align=True output")
print(out)
print()
model = TestModel(align=False)
out = model(x)
print( "pytorch align=False output")
print(out)
print()

torch.onnx.export(model, x, "test_model.onnx")

ort_session = onnxruntime.InferenceSession("test_model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX align=False output")
print(ort_outs)

def patch_interpolate_opset10():
    """Patch interpolate in opset10
    """
    import torch.onnx.symbolic_helper as sym_help
    import torch.onnx.symbolic_opset10

    # noinspection PyProtectedMember
    def _interpolate_size_to_scales(g, input, output_size, dim):
        output_size = sym_help._maybe_get_const(output_size, 'is')
        if sym_help._is_value(output_size):
            offset = 2
            offsets = g.op("Constant", value_t=torch.ones(offset))
            dividend = g.op("Cast", output_size, to_i=sym_help.cast_pytorch_to_onnx["Float"])
            divisor = sym_help._slice_helper(g, g.op("Shape", input), axes=[0], ends=[dim], starts=[offset])
            divisor = g.op("Cast", divisor, to_i=sym_help.cast_pytorch_to_onnx["Float"])
            scale_dims = g.op("Div", dividend, divisor)
            scales = g.op("Concat", offsets, scale_dims, axis_i=0)
        else:
            scales_constant = [1. if i < 2 else
                               float(output_size[-(dim - i)]) / float(input.type().sizes()[-(dim - i)])
                               for i in range(0, dim)]
            scales = g.op("Constant", value_t=torch.tensor(scales_constant))
        return scales

    # noinspection PyProtectedMember
    def _interpolate(name, dim, interpolate_mode):
        # noinspection PyShadowingBuiltins, PyProtectedMember
        def symbolic_fn(g, input, output_size, align_corners=None):
            align_corners = sym_help._maybe_get_scalar(align_corners)
            if align_corners:
                return Exception(name, "align_corners == True")
            scales = _interpolate_size_to_scales(g, input, output_size, dim)
            return g.op("Resize", input, scales, mode_s=interpolate_mode)
        return symbolic_fn

    torch.onnx.symbolic_opset10.upsample_nearest1d = _interpolate('upsample_nearest1d', 3, "nearest")
    torch.onnx.symbolic_opset10.upsample_nearest2d = _interpolate('upsample_nearest2d', 4, "nearest")
    torch.onnx.symbolic_opset10.upsample_nearest3d = _interpolate('upsample_nearest3d', 5, "nearest")
    torch.onnx.symbolic_opset10.upsample_linear1d = _interpolate('upsample_linear1d', 3, "linear")
    torch.onnx.symbolic_opset10.upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, "linear")
    torch.onnx.symbolic_opset10.upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, "linear")

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

print(torch.__version__)
print(onnx.__version__)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x


torch_model = TestModel()
dummy_input = torch.randn(1, 3, 256, 256)

torch_out = torch.onnx.export(torch_model, dummy_input, 'model.onnx', verbose=True, opset_version=11)

onnx_model = onnx.load('model.onnx')
print(onnx_model)
onnx.checker.check_model(onnx_model)

import torch
print("torch version:", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

torch_model = TestModel()
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(torch_model, dummy_input, "/dev/null", verbose=True, opset_version=11)

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime

print(f">>>>> Pytorch version: {torch.__version__}")
print(f">>>>> ONNX Runtime version: {onnxruntime.__version__}\n")

class TestModel(nn.Module):
    def __init__(self, align=False):
        super(TestModel, self).__init__()
        self.align=align
    def forward(self, x):
        x = F.interpolate(x, (4,4), mode='bilinear', align_corners=self.align)
        return x

x = torch.tensor([[0.,1.],[2.,3.]]).view([1,1,2,2])
model = TestModel(align=True)
out = model(x)

print("matrix to be interpolated :")
print(x)
print()
print("-"*80)

print( "pytorch align=True output")
print(out)
print()

torch.onnx.export(model, x, "test_model.onnx", opset_version=11)
ort_session = onnxruntime.InferenceSession("test_model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX align=True output")
print(ort_outs)
print(f">>>>> Are closely the same? --> {torch.allclose(out, torch.from_numpy(ort_outs[0]))}")
print("-"*80)
print()

model = TestModel(align=False)
out = model(x)
print( "pytorch align=False output")
print(out)
print()


torch.onnx.export(model, x, "test_model.onnx", opset_version=11)

ort_session = onnxruntime.InferenceSession("test_model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX align=False output")
print(ort_outs)
print(f">>>>> Are closely the same? --> {torch.allclose(out, torch.from_numpy(ort_outs[0]))}")
print("-"*80)
print()