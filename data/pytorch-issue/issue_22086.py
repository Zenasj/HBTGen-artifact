import torch.nn as nn

import torch
import torch.nn.functional as F
import scipy.stats
import numpy as np

m = 10
n = 10000
replace = False

#################################################
# Comparing Multinomial CPU vs Multinomial CUDA #
#################################################

def check_for_k(k):

	multinomial_cuda_samples = []
	multinomial_cpu_samples = []

	weights = torch.rand(m)

	for _ in range(n):
		multinomial_cuda_samples += torch.multinomial(
							  weights.cuda(),
							  k,
							  replace
							).cpu().numpy().tolist()

		multinomial_cpu_samples += torch.multinomial(
							  weights.cpu(),
							  k,
							  replace
							).cpu().numpy().tolist()


	_, multinomial_cuda_dist = np.unique(multinomial_cuda_samples, return_counts=True)
	_, multinomial_cpu_dist = np.unique(multinomial_cpu_samples, return_counts=True)

	_, p = scipy.stats.chisquare(multinomial_cuda_dist, multinomial_cpu_dist)
	print("Chi-Squared Test p-value {:.3f}".format(p))

	multinomial_cuda_dist = torch.Tensor(multinomial_cuda_dist) / (n * k)
	multinomial_cpu_dist = torch.Tensor(multinomial_cpu_dist) / (n * k)

	print("KL Divergence {:.3f}".format(F.kl_div(
						multinomial_cpu_dist.log(), 
						multinomial_cuda_dist, 
						reduction='sum'
						)
					)
	)


print("CHECKING for k = 1")
check_for_k(1)
print("CHECKING for k = 5")
check_for_k(5)