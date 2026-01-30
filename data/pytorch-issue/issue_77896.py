import torch

self = torch.full((1,), 1.5e+300, dtype=torch.float64, requires_grad=False)
observer_on = torch.full((), -1, dtype=torch.float64, requires_grad=False)
fake_quant_on = torch.full((), -1, dtype=torch.float64, requires_grad=False)
running_min = torch.full((0, 0, 9, 15, 0, 0, 0, 0, 0, 8,), 1, dtype=torch.int64, requires_grad=False)
running_max = torch.full((1,), 0.5, dtype=torch.float64, requires_grad=False)
scale = torch.full((1, 1, 4, 4,), 1, dtype=torch.float32, requires_grad=False)
zero_point = torch.full((1, 1, 4, 4,), 1, dtype=torch.float64, requires_grad=False)
averaging_const = 0
quant_min = 0
quant_max = 0
ch_axis = 1250999896764
per_row_fake_quant = True
symmetric_quant = True
torch.fused_moving_avg_obs_fake_quant(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant)