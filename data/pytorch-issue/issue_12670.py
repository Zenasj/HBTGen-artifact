import torch.nn as nn

import torch
IGNORE_IDX=  -1
print("PyTorch version: {}".format(torch.__version__))
xe = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_IDX, reduction='none')
input = torch.randn(3, 5, requires_grad=True)
print("Input: {}".format(input))

#### test 1:
target = torch.empty(3, dtype=torch.long).random_(5)
target[2] = -4      # should be an error
print("Target: {}".format(target))
loss = xe(input, target)
print("Loss before reduction: {}".format(loss))
output=torch.mean(loss)
print("Loss after reduction: {}".format(output))
output.backward()
print('-'*20)

#### test 2:
target = torch.empty(3, dtype=torch.long).random_(5)
target[2] = IGNORE_IDX      # should be an expected case
print("Target: {}".format(target))
loss = xe(input, target)
print("Loss before reduction: {}".format(loss))
output=torch.mean(loss)
print("Loss after reduction: {}".format(output))
output.backward()

### test 3: run test 2 again
#### test 2:
target = torch.empty(3, dtype=torch.long).random_(5)
target[2] = IGNORE_IDX      # should be an expected case
print("Target: {}".format(target))
loss = xe(input, target)
print("Loss before reduction: {}".format(loss))
output=torch.mean(loss)
print("Loss after reduction: {}".format(output))
output.backward()