import torch

torch.tensor(0.123).log2()
# tensor(-3.0233)
# then applying round, this becomes
# tensor(-3.)

2 ** (torch.tensor(0.123).log2().round()) == 0.125

(2 ** torch.tensor(0.123).log2()).round() == tensor(0.)