import torch.nn as nn

conv = torch.nn.Conv2d(nChans, nChans, kernel_size=kernelSize, padding=(n, 3))
x = conv(inputData)

import torch

nChans = 4
kernelSize = 3
samplesPerBatch = 6
inputShape = [samplesPerBatch, nChans, 32, 32]

inputData = torch.ones(*inputShape, dtype=torch.float32)


def run_conv(padding):
    conv = torch.nn.Conv2d(
        nChans, nChans, kernel_size=kernelSize, padding=padding)
    x = conv(inputData)


# These are all ok
print('Trying conv with padding (1, n)'
      ' where n is all values from 0 to 99 excluding 3')
for i in range(100):
    if i != 3:
        run_conv((1, i))

# This is ok
print('Trying conv with padding (3, 1)')
run_conv((3, 1))

# This one causes an assertion
print('Trying conv with padding (1, 3)')
run_conv((1, 3))