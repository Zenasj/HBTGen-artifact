import torch

input_s = torch.zeros((1, 500), dtype=torch.long).to('cuda')
input_names = ["x"] # specify the name of the input tensor
output_names = ["out"] # specify the name of the output tensor
dynamic_axes = {"x":{0: "batch_size"}, "out":{0: "batch_size"}}
torch.onnx.export(model, input_s, 'model_test.onnx', input_names=input_names, output_names=output_names)