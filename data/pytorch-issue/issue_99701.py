import torch.nn as nn
import random

import torch
from torch import nn
import onnxruntime
import numpy as np

# using pytorch native Transformer
model = nn.Transformer(
    d_model=512,
    nhead= 8,
    num_encoder_layers = 6,
    num_decoder_layers= 6,
    dim_feedforward= 2048,
    dropout= 0.1,
    batch_first=True,
    norm_first=True
)

model.eval()
model.cuda()

# onnx export
with torch.inference_mode(), torch.cuda.amp.autocast():
    torch.onnx.export(
        model=model, 
        args=(
            torch.randn(1, 30, 512, device="cuda"),
            torch.randn(1, 30, 512, device="cuda"),
        ), 
        f="test.onnx", 
        verbose=False, 
        input_names=["tgt", "memory"], 
        output_names=[ "out" ],
        do_constant_folding=True,
        export_params=True,
        dynamic_axes={
            "tgt": {0: 'batch_size', 1: "enc_seq_length"},        # variable length axes
            "memory": {0: 'batch_size', 1: "dec_seq_length"},
            "out": {0: 'batch_size', 1: "out_seq_length"}
        },

    )

assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
device_name = 'gpu'

sess_options = onnxruntime.SessionOptions()
dec_session = onnxruntime.InferenceSession(
    "test.onnx", 
    sess_options,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# the following works
mem = np.random.rand(1, 30, 512).astype(np.float32)
tgt = np.random.rand(1, 30, 512).astype(np.float32)
ort_inputs = {"memory": mem, "tgt": tgt}
ort_outs = dec_session.run(None, ort_inputs)
out = ort_outs[0]
print("this works!")

# the following also works
mem = np.random.rand(5, 30, 512).astype(np.float32)
tgt = np.random.rand(5, 30, 512).astype(np.float32)
ort_inputs = {"memory": mem, "tgt": tgt}
ort_outs = dec_session.run(None, ort_inputs)
out = ort_outs[0]
print("this also works!")

# the following does not work
mem = np.random.rand(1, 20, 512).astype(np.float32)
tgt = np.random.rand(1, 20, 512).astype(np.float32)
ort_inputs = {"memory": mem, "tgt": tgt}
ort_outs = dec_session.run(None, ort_inputs)
out = ort_outs[0]
print("this also also works!")