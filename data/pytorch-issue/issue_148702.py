import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchaudio
import torch

# define a pytorch model
class SpecMaker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = torchvision.transforms.Compose(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=512,
                    win_length=512,
                    hop_length=256,
                ),
                torchaudio.transforms.AmplitudeToDB(top_db=100),
            ]
        )

    def forward(self, x):
        return self.transforms(x)


specmodel = SpecMaker()
input = torch.rand(32000 * 10)
spec = specmodel(input)
input_batch = torch.stack([input, input])

spec_batch = specmodel(input_batch) # just testing pytorch model works as expected
assert spec_batch.shape== torch.Size([2, 257, 1251])

exported_program = torch.export.export(
    specmodel,
    (input_batch,),
    dynamic_shapes=({0: torch.export.Dim.AUTO},),
    strict=False
)

print(f" before decompose: {exported_program}")
#  before decompose: ExportedProgram:
#     class GraphModule(torch.nn.Module):
#         def forward(self, c_lifted_tensor_0: "f32[512]", x: "f32[s0, 320000]"):
#              # 
#             sym_size_int_2: "Sym(s0)" = torch.ops.aten.sym_size.int(x, 0)
            
#              # File: /home/titaiwang/audio/src/torchaudio/transforms/_transforms.py:110 in forward, code: return F.spectrogram(
#             reshape: "f32[s0, 320000]" = torch.ops.aten.reshape.default(x, [-1, 320000]);  x = None
#             view: "f32[1, s0, 320000]" = torch.ops.aten.view.default(reshape, [1, sym_size_int_2, 320000]);  reshape = None
#             pad: "f32[1, s0, 320512]" = torch.ops.aten.pad.default(view, [256, 256], 'reflect');  view = None
#             view_1: "f32[s0, 320512]" = torch.ops.aten.view.default(pad, [sym_size_int_2, 320512]);  pad = None
#             stft: "c64[s0, 257, 1251]" = torch.ops.aten.stft.default(view_1, 512, 256, 512, c_lifted_tensor_0, False, True, True);  view_1 = c_lifted_tensor_0 = None
#             reshape_1: "c64[s0, 257, 1251]" = torch.ops.aten.reshape.default(stft, [sym_size_int_2, 257, 1251]);  stft = None
#             abs_1: "f32[s0, 257, 1251]" = torch.ops.aten.abs.default(reshape_1);  reshape_1 = None
#             pow_1: "f32[s0, 257, 1251]" = torch.ops.aten.pow.Tensor_Scalar(abs_1, 2.0);  abs_1 = None
            
#              # File: /home/titaiwang/audio/src/torchaudio/transforms/_transforms.py:345 in forward, code: return F.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)
#             clamp: "f32[s0, 257, 1251]" = torch.ops.aten.clamp.default(pow_1, 1e-10);  pow_1 = None
#             log10: "f32[s0, 257, 1251]" = torch.ops.aten.log10.default(clamp);  clamp = None
#             mul: "f32[s0, 257, 1251]" = torch.ops.aten.mul.Tensor(log10, 10.0);  log10 = None
#             sub_: "f32[s0, 257, 1251]" = torch.ops.aten.sub_.Tensor(mul, 0.0);  mul = None
#             reshape_2: "f32[1, s0, 257, 1251]" = torch.ops.aten.reshape.default(sub_, [-1, sym_size_int_2, 257, 1251]);  sub_ = None
#             amax: "f32[1]" = torch.ops.aten.amax.default(reshape_2, [-3, -2, -1])
#             sub: "f32[1]" = torch.ops.aten.sub.Tensor(amax, 100);  amax = None
#             view_2: "f32[1, 1, 1, 1]" = torch.ops.aten.view.default(sub, [-1, 1, 1, 1]);  sub = None
#             max_1: "f32[1, s0, 257, 1251]" = torch.ops.aten.max.other(reshape_2, view_2);  reshape_2 = view_2 = None
#             reshape_3: "f32[s0, 257, 1251]" = torch.ops.aten.reshape.default(max_1, [sym_size_int_2, 257, 1251]);  max_1 = sym_size_int_2 = None
#             return (reshape_3,)
            
# Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.CONSTANT_TENSOR: 4>, arg=TensorArgument(name='c_lifted_tensor_0'), target='lifted_tensor_0', persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='x'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='reshape_3'), target=None)])
# Range constraints: {s0: VR[2, int_oo]}
exported_program = exported_program.run_decompositions(decomp_table=None)
print(f" after decompose: {exported_program}")
#  after decompose: ExportedProgram:
#     class GraphModule(torch.nn.Module):
#         def forward(self, c_lifted_tensor_0: "f32[512]", x: "f32[2, 320000]"):
#              # File: /home/titaiwang/audio/src/torchaudio/transforms/_transforms.py:110 in forward, code: return F.spectrogram(
#             view: "f32[2, 320000]" = torch.ops.aten.view.default(x, [-1, 320000]);  x = None
#             view_1: "f32[1, 2, 320000]" = torch.ops.aten.view.default(view, [1, 2, 320000]);  view = None
#             arange: "i64[320512]" = torch.ops.aten.arange.start_step(-256, 320256, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
#             abs_1: "i64[320512]" = torch.ops.aten.abs.default(arange);  arange = None
#             sub: "i64[320512]" = torch.ops.aten.sub.Tensor(319999, abs_1);  abs_1 = None
#             abs_2: "i64[320512]" = torch.ops.aten.abs.default(sub);  sub = None
#             sub_1: "i64[320512]" = torch.ops.aten.sub.Tensor(319999, abs_2);  abs_2 = None
#             index: "f32[1, 2, 320512]" = torch.ops.aten.index.Tensor(view_1, [None, None, sub_1]);  view_1 = sub_1 = None
#             view_2: "f32[2, 320512]" = torch.ops.aten.view.default(index, [2, 320512]);  index = None
#             unfold: "f32[2, 1251, 512]" = torch.ops.aten.unfold.default(view_2, -1, 512, 256);  view_2 = None
#             mul: "f32[2, 1251, 512]" = torch.ops.aten.mul.Tensor(unfold, c_lifted_tensor_0);  unfold = c_lifted_tensor_0 = None
#             _fft_r2c: "c64[2, 1251, 257]" = torch.ops.aten._fft_r2c.default(mul, [2], 0, True);  mul = None
#             permute: "c64[2, 257, 1251]" = torch.ops.aten.permute.default(_fft_r2c, [0, 2, 1]);  _fft_r2c = None
#             view_3: "c64[2, 257, 1251]" = torch.ops.aten.view.default(permute, [2, 257, 1251]);  permute = None
#             abs_3: "f32[2, 257, 1251]" = torch.ops.aten.abs.default(view_3);  view_3 = None
#             pow_1: "f32[2, 257, 1251]" = torch.ops.aten.pow.Tensor_Scalar(abs_3, 2.0);  abs_3 = None
            
#              # File: /home/titaiwang/audio/src/torchaudio/transforms/_transforms.py:345 in forward, code: return F.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)
#             clamp: "f32[2, 257, 1251]" = torch.ops.aten.clamp.default(pow_1, 1e-10);  pow_1 = None
#             log10: "f32[2, 257, 1251]" = torch.ops.aten.log10.default(clamp);  clamp = None
#             mul_1: "f32[2, 257, 1251]" = torch.ops.aten.mul.Tensor(log10, 10.0);  log10 = None
#             sub_2: "f32[2, 257, 1251]" = torch.ops.aten.sub.Tensor(mul_1, 0.0);  mul_1 = None
#             view_5: "f32[1, 2, 257, 1251]" = torch.ops.aten.view.default(sub_2, [1, 2, 257, 1251]);  sub_2 = None
#             amax: "f32[1]" = torch.ops.aten.amax.default(view_5, [-3, -2, -1])
#             sub_3: "f32[1]" = torch.ops.aten.sub.Tensor(amax, 100);  amax = None
#             view_6: "f32[1, 1, 1, 1]" = torch.ops.aten.view.default(sub_3, [-1, 1, 1, 1]);  sub_3 = None
#             maximum: "f32[1, 2, 257, 1251]" = torch.ops.aten.maximum.default(view_5, view_6);  view_5 = view_6 = None
#             view_7: "f32[2, 257, 1251]" = torch.ops.aten.view.default(maximum, [2, 257, 1251]);  maximum = None
#             return (view_7,)
            
# Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.CONSTANT_TENSOR: 4>, arg=TensorArgument(name='c_lifted_tensor_0'), target='lifted_tensor_0', persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='x'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='view_7'), target=None)])
# Range constraints: {}