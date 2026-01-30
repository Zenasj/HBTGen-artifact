import torch.nn as nn

class IdxModel(torch.nn.Module):
    def forward(self, x):
        b = torch.ones(x.shape[1:])
        i = 1
        x[i] = b
        x += 1. # so that onnx model is not a no-op
        return x

3
class IdxModelScatter(torch.nn.Module):
    def forward(self, x):
        b = torch.ones((x.shape[1:]))
        i = 1
        idx = torch.tensor([i], dtype=torch.int64).expand(x.shape[1:]).unsqueeze(0)
        x.scatter_(0, idx, b.unsqueeze(0))
        x += 1. 
        return x

3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import onnx
import torch 
from torch import Tensor
import onnxruntime as ort
from copy import deepcopy

def onnx_export_and_check_equiv(model):
    """Exports onnx inplace indexing model and returns ort and PyTorch equivalence."""
    input_size = 2
    seq_len = 3
    batch = 2
    INPUT_NAMES = ["input"]
    OUTPUT_NAMES = ["out"]
    DYNAMIC_AXES = {
        "input": {0: "seq_len", 1: "batch"},
        "out": {0: "seq_len", 1: "batch"},

    }
    fp = 'idx.onnx'

    # gen args for torch and ort session
    args = torch.zeros(seq_len, batch, input_size),
    args_np = [x.numpy() for x in deepcopy(args)]
    dict_args = {k: args_np[idx] for idx, k in enumerate(INPUT_NAMES)}


    example_outputs = model(*deepcopy(args))

    torch.onnx.export(
                model,
                args,
                fp,
                export_params=True,
                verbose=False,
                example_outputs=example_outputs,
                dynamic_axes=DYNAMIC_AXES,
                input_names=INPUT_NAMES,
                output_names=OUTPUT_NAMES,
                opset_version=11,
            )

    onnx_model = onnx.load(fp)
    onnx.checker.check_model(onnx_model)

    # start ort session 
    ort_session = ort.InferenceSession(fp)

    outputs = ort_session.run(None, dict_args)
    return torch.allclose(torch.tensor(outputs), example_outputs, atol=1e-4, rtol=1e-2)

model = IdxModel()
assert not onnx_export_and_check_equiv(model)

model = IdxModelScatter()
assert  onnx_export_and_check_equiv(model)

3
model = torch.jit.script(IdxModelScatter())
assert  onnx_export_and_check_equiv(model)