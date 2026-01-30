python
import torch

data = torch.tensor([8, 7, 3, 4, 5, 1, 7, 2], dtype=torch.float32)
lengths = torch.tensor([2, 3, 3], dtype=torch.int64)

def simple_model(data, lengths):
    return torch.segment_reduce(data, reduce="max", lengths=lengths)


compiled_model = torch.compile(simple_model, backend="inductor")
compiled_model(data, lengths)