import torch as tt
import torch.nn as nn

# Random image
image = tt.randn(1, 1, 512, 512)

# Conv2d with kernel size of 3 results in different output if conv_layer has gradients enabled or not
conv_layer_3 = nn.Conv2d(1, 3, 3)
test_3_a = conv_layer_3(image)
conv_layer_3.requires_grad_(False)
test_3_b = conv_layer_3(image)
print(tt.equal(test_3_a, test_3_b)) # False. The differences are between 1e-5 and 1e-8

# Conv2d with kernel size of 5 does not show the same behaviour
conv_layer_5 = nn.Conv2d(1, 3, 5)
test_5_a = conv_layer_5(image)
conv_layer_5.requires_grad_(False)
test_5_b = conv_layer_5(image)
print(tt.equal(test_5_a, test_5_b)) # True

import torch
x = torch.randn(1, 512, 512)
m = torch.nn.Conv2d(1, 1, 3)
o1 = m(x)
with torch.no_grad():
    o2 = m(x)
print((o1-o2).abs().max())