import torch

@skipIf(not torch.cuda.is_available())
def test_gpu():
    ...

@skipIf(torch.cuda.device_count() < 2)
def test_distributed():
    ...

@filterwarnings("ignore:CUDA initialization.*:UserWarning")
@skipIf(not torch.cuda.is_available())
def test_gpu():
    ...

@filterwarnings("ignore:CUDA initialization.*:UserWarning")
@skipIf(torch.cuda.device_count() < 2)
def test_distributed():
    ...

@skipIf(not torch.cuda.is_available())
def test_gpu():
    ...

@skipIf(not torch.cuda.is_available() or torch.cuda.device_count() < 2)
def test_distributed():
    ...