import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)  # ERROR
    print(x)
else:
    print ("MPS device not found.")

torch.ones(1)  # OK: tensor([1.])
torch.ones(1).to('mps')  # ERROR
torch.zeros(1).to('mps') # OK: tensor([0.], device='mps:0')
torch.randn(1).to('mps')  # ERROR
torch.rand(1).to('mps')  # ERROR