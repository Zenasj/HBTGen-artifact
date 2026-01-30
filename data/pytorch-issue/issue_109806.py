import onnxruntime as ort
import torch


model_args = torch.load("model_args.pt")
loaded = torch.jit.load("traced_model.pt")


torch.onnx.export(loaded, model_args, "./model.onnx", opset_version=16)

ort.InferenceSession("./model.onnx", providers=["CPUExecutionProvider"])