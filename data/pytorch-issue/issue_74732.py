import torch

from torch.onnx import verification

verification.verify(
    model,
    input_tuple,
    opset_version=12,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
)

verification.find_mismatch(
    model, input_tuple, opset_version=12, do_constant_folding=True
)