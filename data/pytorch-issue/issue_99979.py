import torch

def to_onnx_zdim(model,onnx_name):
      dummy_input=torch.randn(1,1,512,device='cuda')
      torch.onnx.export(model,dummy_input,onnx_name,verbose=True)