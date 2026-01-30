import torch

model = torch.jit.script(MyModel())
example_outputs = model(x)
torch.onnx.export(model, (x,), "model.onnx", example_outputs=example_outputs)

model = torch.jit.script(MyModel())
torch.onnx.export(model, (x,), "model.onnx")