# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import trainer
import torch
from torch import jit

def load_model():
    filename_model = '/path/to/pretrained/model'
    model = torch.load(filename_model, map_location='cpu').eval()

    return model

model = load_model()
# My model contains input dependent control flow, so I decided to convert my model to TorchScript first, then to ONNX project.
model.onset_stack[1] = torch.jit.script(model.onset_stack[1])
model.offset_stack[1] = torch.jit.script(model.offset_stack[1])
model.combined_stack[0] = torch.jit.script(model.combined_stack[0])
model = torch.jit.trace(model, torch.rand([1,1,100,229]))

output_dynamic_ax = {1: 'frame_num'}
torch.onnx.export(
model,
torch.rand([1, 1, 100, 229]),
"piano_trans.onnx",
verbose=True,
input_names='mel_spec',
output_names=['onset_pred', 'offset_pred', 'activation_pred', 'frame_pred'],
example_outputs=model(torch.rand([1, 1, 100, 229])),
dynamic_axes={'mel_spec': {2: 'frame_num'}, 'onset_pred': output_dynamic_ax, 
'offset_pred': output_dynamic_ax, 'activation_pred': output_dynamic_ax, 'frame_pred': output_dynamic_ax}
)