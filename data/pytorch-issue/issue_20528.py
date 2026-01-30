import torch.nn as nn

python
import torch
from concurrent.futures import ThreadPoolExecutor

# dummy data and network
input_data = torch.arange(10, dtype=torch.float32)
net = torch.nn.Sequential(
    torch.nn.Linear(10, 1)
)

with ThreadPoolExecutor(2) as ex:
    with torch.no_grad():
        # no_grad is working
        assert net(input_data).grad_fn is None
        
        # Should fail because of the "not" <-------------------------------
        assert list(ex.map(net, [input_data]))[0].grad_fn is not None
        
        # no_grad is working
        assert list(ex.map(torch.no_grad()(net), [input_data]))[0].grad_fn is None
        
        # no_grad is working
        assert net(input_data).grad_fn is None
        
    # Can calculate the gradient
    assert net(input_data).grad_fn is not None