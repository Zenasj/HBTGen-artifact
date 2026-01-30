import torch.nn as nn

import torch
import onnx
from onnxsim import simplify

class simple_padding(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        pad_shape = [1, 1, 1, 1]
   
        # return torch.nn.functional.pad(x, pad_shape)
        return torch.nn.functional.pad(x, pad_shape, value=0)

def padding_export_check():

    onnx_save_path = "padding.onnx"
    simplified_save_path = "opt_paddingonnx"

    model = simple_padding()

    dummy_input = torch.rand(3, 224, 224, device='cpu')

    torch.onnx.export(model,
                      args=(dummy_input),
                      f=onnx_save_path,
                      input_names=["input"],
                      opset_version=11)

    # onnx simplify
    saved_onnx = onnx.load(onnx_save_path)
    onnx_simplified, check = simplify(saved_onnx)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(onnx_simplified), simplified_save_path)