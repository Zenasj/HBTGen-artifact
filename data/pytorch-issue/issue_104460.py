if noisequant:
  y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor, "noise")
  y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor, "ste")
else:
  y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                            "ste") + means_non_anchor
  y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                            "ste") + means_non_anchor

import torch
import os
import math
import torch.nn as nn
import compressai
from compressai.zoo import load_state_dict
from Network import TestModel

import onnxscript
from onnxscript.onnx_opset import opset18 as op
from torch.onnx._internal import jit_utils
import math

opset_version = 14
custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=1)
@onnxscript.script(custom_opset)
def erfc(x):
    y=op.Sub(1, op.Erf(x))
    return y

def custom_erfc(g: jit_utils.GraphContext, x):
    return g.onnxscript_op(erfc).setType(x.type())

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::erfc",
    symbolic_fn=custom_erfc,
    opset_version=opset_version,
) 


def load_pth_model(paths):
    state_dict = load_state_dict(torch.load(paths))
    model_cls = TestModel()
    model = model_cls.from_state_dict(state_dict).eval()
    return model

def write_onxx_model(model, paths, opset_version):
    opset_version = opset_version
    onnx_file_name = (paths.split('/')[-1]).split('.')[0]
    print(onnx_file_name)
    dummy_input = torch.rand(1, 3, 512, 768) 
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_file_name+".onnx",
                      export_params=True,
                      opset_version=opset_version,
                      input_names = ['input'],
                      output_names = ['output']) # , verbose=True   
    
paths = '../ELiC-ReImplemetation/ELIC_0450_ft_3980_Plateau.pth.tar'
model = load_pth_model(paths) 
'''
# check
torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
    model, args = torch.rand(1, 3, 512, 768), opset_version=opset_version
)
print(set(unconvertible_ops))
'''
write_onxx_model(model, paths, opset_version)