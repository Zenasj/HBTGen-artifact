import torch

tensor  = torch.LongTensor(...)
tensor.new(1, device=torch.device('mps'))

unfinished_sequences = torch.ones(input_ids.shape[0], device=input_ids.device)