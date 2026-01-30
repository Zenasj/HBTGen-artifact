import torch

net = Net().to(torch.bfloat16)
torch.save(net.state_dict(), "test.pt")
net.load_state_dict(torch.load("test.pt", weights_only=True))