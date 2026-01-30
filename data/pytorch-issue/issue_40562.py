import torch.nn as nn

from torch import nn
import torch
import math


def get_mem_usage():
    return convert_size(torch.cuda.max_memory_allocated(device='cuda'))

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class MemoryModel(nn.Module):

    def __init__(self):

        super(MemoryModel, self).__init__()

        self.model = nn.Linear(128, 128)

    def forward(self, input_ids):
        print('before', get_mem_usage())
        output = self.model(input_ids)
        print('After first execution', get_mem_usage())        
        self.model(input_ids)
        print('After second execution: Again more memory is allocated... why?', get_mem_usage())
        output = self.model(input_ids)
        print('Upcoming executions do not make a difference...', get_mem_usage())
        self.model(input_ids)
        print('Upcoming executions do not make a difference...', get_mem_usage())


print('emtpy', get_mem_usage())
batch_size = 1024
device = torch.device('cuda')
input_ids = torch.tensor([[0.1]*128]*batch_size)
input_ids = input_ids.to(device)
print('added input ids', get_mem_usage())
model = MemoryModel()
model.to(device)
print('added model', get_mem_usage())
with torch.no_grad(): 
    model(input_ids)