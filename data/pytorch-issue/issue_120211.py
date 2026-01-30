import torch.nn as nn

py
import torch
import torch.nn.functional


def reproducer(radius: int):
    image = torch.rand([1, 1024, 1024, 3], dtype=torch.float32, device=torch.device('cpu'))
    image = image.permute(0, 3, 1, 2)
    # creating the tensor above with the appropriate shape directly does not reproduce the issue
    #image = torch.rand([1, 3, 1024, 1024])

    kernel_x = torch.zeros([3, 1, 1, radius * 2 + 1], device=image.device)

    image = torch.nn.functional.conv2d(image, kernel_x, groups=image.shape[-3])


for i in range(0, 128):
    print(i)
    reproducer(radius=i)