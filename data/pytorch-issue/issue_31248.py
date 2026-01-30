import torch
import torch.nn as nn

class KL_Loss(nn.Module):
	def __init__(self):
		super(KL_Loss, self).__init__()

	def forward(self, p_mu, p_log_var, q_mu, q_log_var):
		# [batch_size,d]  d is the dimension of my multivariate gaussian distribution
		assert q_mu.size() == p_mu.size()
		assert q_log_var.size() == p_log_var.size()

		# suppose the Neural Network esitimate log_var 
		q_var = torch.diag_embed(torch.exp(q_log_var))
		p_var = torch.diag_embed(torch.exp(p_log_var))
		
		p = torch.distributions.MultivariateNormal(p_mu, p_var)
		q = torch.distributions.MultivariateNormal(q_mu, q_var)
		kl_loss = torch.distributions.kl_divergence(p, q).mean()

		return kl_loss