from typing import List
import torch
import torch._dynamo
import torch._inductor
from torch._inductor import config
import logging
from torchvision import models
import math

# torch._dynamo.config.log_level = logging.DEBUG
# torch._dynamo.config.verbose = True
# torch._inductor.config.debug = True

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

batch_size = 4096
# batch_size = 1024
device = "cuda"

resnet18 = resnet18.eval().to(device)
opt_resnet18 = torch._dynamo.optimize("inductor")(resnet18)
# opt_resnet18 = resnet18

count = 0
while batch_size >= 500 and count < 5:
    try:
        print("batch size = ", batch_size)
        print("start: ", convert_size(torch.cuda.memory_allocated()))
        input = torch.randn((batch_size, 3, 224, 224)).to(device)
        output = opt_resnet18(input)
        print(output.shape)
    except RuntimeError as e:
        print(e)
        print("in runtime error: ", convert_size(torch.cuda.memory_allocated()))

    print("end: ", convert_size(torch.cuda.memory_allocated()))
    count += 1
    batch_size = int(batch_size / 2)