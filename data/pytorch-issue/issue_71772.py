import torch
import torch.nn as nn

m = torch.nn.Bilinear(20, 30, 40).to("cuda")

input1 = torch.randn(128, 20).to("cuda")
input2 = torch.randn(128, 30).to("cuda")

torch.onnx.export(m,               # model being run
                  (
                  input1,
                  input2
                  ),                         # model input (or a tuple for multiple inputs)
                  "Bilinear.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  verbose = False,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input1',
                                'input2'],   # the model's input names
                  output_names = ['hidden_state'])