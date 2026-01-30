import sys
import torch
def handler(event, context):
    print(torch.__version__, torch.__config__.show())
    torch.set_num_threads(8)
    a=torch.rand(100, 100, 100); b=torch.rand(100,100, 100)
    for i in range(10):
        torch.bmm(a,b).sum()

    return torch.__version__