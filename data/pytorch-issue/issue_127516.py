import torch.nn as nn

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq

class M(torch.nn.Module):
    def __init__(
        self,
        add_fn,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.add_fn = add_fn
        self.relu = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(4, 4)
        self.linear4 = torch.nn.Linear(4, 4)
        self.add_fn2 = add_fn
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        tmp = self.add_fn(x1, x2)
        return tmp
        # tmp1 = self.linear3(tmp)
        # tmp2 = self.linear4(tmp)
        # res = self.add_fn2(tmp1, tmp2)
        # return res

add_fn = lambda x, y: x.add_(y)
mod = M(add_fn)
x = torch.randn((4, 4), dtype=torch.float32, requires_grad=False).add(1)
inputs = (x,)

with torch.no_grad():
    export_model = capture_pre_autograd_graph(mod, inputs)
    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        xiq.get_default_x86_inductor_quantization_config()
    )

    prepare_model = prepare_pt2e(export_model, quantizer)
    prepare_model(*inputs)
    convert_model = convert_pt2e(prepare_model)
    torch.ao.quantization.move_exported_model_to_eval(convert_model)
    print(convert_model)

    out_ref = convert_model(*inputs)
    compiled_model = torch.compile(convert_model)
    compiled_model(*inputs)
    out = compiled_model(*inputs)
    print(torch.allclose(out_ref, out))
    print(torch.abs(out_ref - out).max())

def call(args):
    arg8_1, = args
    args.clear()
    assert_size_stride(arg8_1, (4, 4), (4, 1))
    buf0 = empty_strided_cpu((4, 4), (4, 1), torch.uint8)
    buf1 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf2 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf3 = empty_strided_cpu((4, ), (1, ), torch.int64)
    cpp_fused_quantize_per_tensor_0(arg8_1, buf0, buf1, buf2, buf3)
    del arg8_1
    buf4 = torch.ops.onednn.qlinear_pointwise.default(buf0, 0.012747906148433685, 82, _frozen_param9, buf2, buf3, buf1, 1.0, 0, torch.float32, 'none', [None], '')
    assert_size_stride(buf4, (4, 4), (4, 1))
    del buf0
    del buf1
    del buf2
    del buf3
    return (buf4, )

def call(args):
    arg8_1, = args
    args.clear()
    assert_size_stride(arg8_1, (4, 4), (4, 1))
    buf0 = empty_strided_cpu((4, 4), (4, 1), torch.uint8)
    buf1 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf2 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf3 = empty_strided_cpu((4, ), (1, ), torch.int64)
    cpp_fused_quantize_per_tensor_0(arg8_1, buf0, buf1, buf2, buf3)
    del arg8_1
    buf4 = torch.ops.onednn.qlinear_pointwise.default(buf0, 0.016188886016607285, 46, _frozen_param9, buf2, buf3, buf1, 1.0, 0, torch.float32, 'none', [None], '')
    assert_size_stride(buf4, (4, 4), (4, 1))
    buf5 = buf2; del buf2  # reuse
    buf6 = buf1; del buf1  # reuse
    buf7 = buf3; del buf3  # reuse
    cpp_fused_1(buf5, buf6, buf7)
    buf8 = torch.ops.onednn.qlinear_pointwise.binary(buf0, 0.016188886016607285, 46, _frozen_param8, buf6, buf7, buf5, 1.0, 0, torch.float32, buf4, 1.0, 0, 'sum', 1.0, 'none', [None], '')
    del buf0
    del buf5
    del buf6
    del buf7
    return (buf4, )