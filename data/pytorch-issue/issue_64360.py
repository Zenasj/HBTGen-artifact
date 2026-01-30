import torch

with enable_python_mode(LoggingTensor):
    z = torch.empty([])
    assert isinstance(z, LoggingTensor)