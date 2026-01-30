import torch

ab = torch.zeros(5, dtype=torch.bool)
af = torch.zeros(5, dtype=torch.float)
ad = torch.zeros(5, dtype=torch.double)

# This fail as expected
torch.where(ab, af, ad)  # RuntimeError: expected scalar type float but found double

# This works as expected
torch.where(ab, ad, 3.0)

# WTF are you doing!
torch.where(ab, af, 3.0)  # RuntimeError: expected scalar type float but found double

torch.where(ab, af, torch.tensor(3.0, dtype=torch.double))

torch.where(ab, af, 3.0)