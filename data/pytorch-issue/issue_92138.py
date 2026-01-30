import torch.nn as nn

import torch
import numpy as np
import random

local_seed = 2022
torch.manual_seed(local_seed) # Set PyTorch seed
np.random.seed(seed=local_seed) # Set Numpy seed
random.seed(local_seed) # Set the Python seed

class SimpleNet(torch.nn.Module):
    def __init__(self, with_bias=False):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, (1, 2), stride=(1, 2), padding=(1, 1), bias=with_bias)
        self.conv2 = torch.nn.Conv2d(2, 2, (1, 2), stride=(1, 2), padding=(1, 1), bias=with_bias)

        self.relu = torch.nn.ReLU()
    def forward(self, x, x2): 
        x2 = self.conv2(x2)       
        x1 = self.relu(torch.add(self.conv1(x), x2))
        return x1

def test_pytorch_quantization():
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    from torch.ao.quantization import QConfigMapping
    import torch.quantization.quantize_fx as quantize_fx
    batch_size = 1
    model = SimpleNet().eval()

    x = torch.rand(batch_size, 2, 10, 7)
    x2 = torch.rand(batch_size, 2, 10, 7)

    example_inputs = (x, x2)
    res_ref = model(x, x2)
    qconfig = QConfig(
            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    backend_config = torch.ao.quantization.backend_config.onednn.get_onednn_backend_config()
    torch.backends.quantized.engine = 'onednn'
    prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs, backend_config=backend_config)
    with torch.no_grad():
        for i in range(1):
            prepared_model(x, x2)
        model = quantize_fx.convert_fx(prepared_model, backend_config=backend_config)
        for i in range(3):
            model(x, x2)

if __name__ == "__main__":
    test_pytorch_quantization()