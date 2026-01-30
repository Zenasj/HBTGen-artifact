import torch
checkpoint = "weights/bb5792a6.pt"
model = load_from_checkpoint(checkpoint, **kwargs).eval()



model_path = f"weights/model.onnx"

dummy_input = torch.randn(1, 3, 32, 128)
torch.onnx.export(model, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['output'], opset_version=14)