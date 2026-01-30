import torch

torch.tensor(1., device='cpu') / torch.tensor([2., 2.], device='cpu')[0][None]
# tensor([0.5000])

torch.tensor(1., device='cpu') / torch.tensor([2., 2.], device='cpu')[1][None]
# tensor([0.5000])

# good
torch.tensor(1., device='mps') / torch.tensor([2., 2.], device='mps')[0][None]
# tensor([0.5000], device='mps:0')

# BAD!
torch.tensor(1., device='mps') / torch.tensor([2., 2.], device='mps')[1][None]
# tensor([inf], device='mps:0')

torch.tensor(1., device='mps') / torch.tensor([2., 2.], device='mps')[1][None].detach()
# tensor([0.5000], device='mps:0')