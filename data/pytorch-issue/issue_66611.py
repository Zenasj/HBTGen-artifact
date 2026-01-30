import torch

torch.onnx.export(MyModel(), (x,), "model.onnx", enable_onnx_checker=False)

try:
	torch.onnx.export(MyModel(), (x,), "model.onnx")
except torch.onnx.CheckerError:
	print("ONNX graph is invalid but the ONNX file is still generated.")