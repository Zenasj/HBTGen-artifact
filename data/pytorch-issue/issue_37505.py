import torch
import torch.nn as nn

torch.manual_seed(1)

# some random input
a = torch.Tensor(1, 2, 1, 1).random_().cuda()

# 2 individual Conv2d layers
conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=False).cuda()
conv2 = nn.Conv2d(1, 1, 3, padding=1, bias=False).cuda()

# Conv2d layer with groups
conv_gr = nn.Conv2d(2, 2, 3, groups=2, padding=1, bias=False).cuda()
# taking weights from the individual layers
conv_gr.weight.data = torch.cat([conv1.weight.data, conv2.weight.data], 0)

# concatenation of the outputs of the 2 individual layers
print(torch.cat([conv1(a[:, 0:1, ...]), conv2(a[:, 1:2, ...])], 0).data)
# out of layer with groups
print(conv_gr(a).data)

tensor([[[[-871647.5000]]],
        [[[-140616.8438]]]], device='cuda:0')
tensor([[[[-871647.5000]],
         [[-140616.8594]]]], device='cuda:0')

tensor([[[[-871647.5000]]],
        [[[-140616.8594]]]])
tensor([[[[-871647.5000]],
         [[-140616.8594]]]])

In [11]: print(torch.cat([conv1(a[:, 0:1, ...]), conv2(a[:, 1:2, ...])], 0).data)
tensor([[[[-871647.5000]]],
        [[[-140616.8594]]]], device='cuda:0')

In [12]: print(conv_gr(a).data)
tensor([[[[-871647.5000]],
         [[-140616.8594]]]], device='cuda:0')