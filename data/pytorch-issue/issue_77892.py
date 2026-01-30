import torch

self = torch.full((1,), 0.5, dtype=torch.float64, requires_grad=False)
observer_on = torch.full((1,), 0.5, dtype=torch.float64, requires_grad=False)
fake_quant_on = torch.full((1,), 1, dtype=torch.float32, requires_grad=False)
running_min = torch.full((5, 5, 5, 5, 5,), 3.5e+35, dtype=torch.float64, requires_grad=False)
running_max = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
scale = torch.full((0, 3, 8, 0, 6, 10, 0, 0, 5, 0,), 0, dtype=torch.float64, requires_grad=False)
zero_point = torch.full((0,), 1, dtype=torch.float32, requires_grad=False)
averaging_const = 0
quant_min = 0
quant_max = 0
ch_axis = 1250999896764
per_row_fake_quant = True
symmetric_quant = True
torch._fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant)