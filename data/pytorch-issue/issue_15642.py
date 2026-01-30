import torch

network = torch.jit.trace(self.model, img_list_torch)
network.save(network_path)

theta_tmp = torch.stack([theta_i[0, 0], theta_i[0, 1]]).unsqueeze(1)
theta = torch.cat([scale_factors, theta_tmp], dim= 1).unsqueeze(0)

theta_tmp = torch.stack([theta_i[0, 0], theta_i[0, 1]])
theta_tmp = theta_tmp.unsqueeze(1)
theta = torch.cat((scale_factors.transpose(0, 1), theta_tmp.transpose(0, 1))).transpose(0, 1)
theta = theta.unsqueeze(0)