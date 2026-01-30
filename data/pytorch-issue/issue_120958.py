3
import torch
import torch.nn as nn
from typing import Optional

# Simplified example of the diffuser lora layer here: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/lora.py#L380
class LoraCompatibleLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = None

    def set_lora_layer(self, lora_layer: Optional[nn.Module]):
        self.lora_layer = lora_layer

    def forward(self, x):
        if self.lora_layer is None:
            return super().forward(x)
        else:
            return super().forward(x) + self.lora_layer(x)


def test(compile):
    net = LoraCompatibleLinear(1, 1)
    nn.init.ones_(net.weight)
    nn.init.zeros_(net.bias)
    lora_layer = nn.Linear(1, 1)
    nn.init.ones_(lora_layer.weight)
    nn.init.zeros_(lora_layer.bias)

    if compile:
        net = torch.compile(net)

    x = torch.ones(1, 1, 1)

    with torch.no_grad():
        # no lora
        y1 = net(x)
        # apply lora
        net.set_lora_layer(lora_layer)
        y2 = net(x)
        # update lora weight
        nn.init.constant_(lora_layer.weight, 2)
        y3 = net(x)

    # expect 1, 2, 3
    return y1, y2, y3


# without torch.compile the code returns expected `1, 2, 3`
print(test(False))

# with torch.compile the code returns unexpected `1, 1, 1`
print(test(True))