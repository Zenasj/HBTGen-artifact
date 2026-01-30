import torch

print(f"Pytorch version: {torch.__version__}")
print(f"Is CUDA available?: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
device = torch.device('cuda')
print(f"A torch tensor: {torch.rand(5).cuda(0)}")
print(f"A torch tensor: {torch.rand(5).cuda(1)}")