import torch

model.load_state_dict('checkpoint.pth')
input = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
input = input.to(device)
# Export the model
torch_out = torch.onnx._export(model,
					   input,                      
					   "model_caff2.onnx", 
					   export_params=True)