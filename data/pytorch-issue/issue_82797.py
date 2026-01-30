import torch, pytest
... if torch.cuda.is_available() else pytest.skip()