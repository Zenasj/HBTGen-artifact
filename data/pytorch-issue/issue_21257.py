import torch.nn as nn

import torch
import scipy.stats
import numpy as np
import torch.nn.functional as F

n = 10000
replace = True
device = 'cpu'

multinomial_alias_samples = []
multinomial_samples = []

weights = torch.Tensor([0.1, 0.6, 0.2, 0.1], device=device)
J, q = torch._multinomial_alias_setup(weights)

for _ in range(n):
	multinomial_alias_samples += torch._multinomial_alias_draw(
							      q, 
							      J, 
							      1
							    ).cpu().numpy().tolist()
	multinomial_samples += torch.multinomial(
						  weights,
						  1,
						  replace
						).cpu().numpy().tolist()

correct_dist =  weights / weights.sum()
correct_dist = correct_dist.to('cpu')
_, multinomial_alias_dist = np.unique(multinomial_alias_samples, return_counts=True)

_, p = scipy.stats.chisquare(multinomial_alias_dist, correct_dist.numpy() * n)
print("[ALIAS] Chi-Squared Test p-value {:.3f}".format(p))
multinomial_alias_dist = torch.Tensor(multinomial_alias_dist) / n
print("[ALIAS] KL Divergence {:.3f}".format(
					    F.kl_div(
						multinomial_alias_dist.log(), 
						correct_dist, 
						reduction='sum')
					    )
      )

_, multinomial_dist = np.unique(multinomial_samples, return_counts=True)
_, p = scipy.stats.chisquare(multinomial_dist, correct_dist.numpy() * n)
print("[NO ALIAS] Chi-Squared Test p-value {:.3f}".format(p))
multinomial_dist = torch.Tensor(multinomial_dist) / n
print("[NO ALIAS] KL Divergence {:.3f}".format(
					    F.kl_div(
						multinomial_dist.log(), 
						correct_dist, 
						reduction='sum')
					    )
      )