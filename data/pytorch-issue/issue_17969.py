import torch

out = torch.utils.checkpoint.checkpoint(run_fn, input_var, None)