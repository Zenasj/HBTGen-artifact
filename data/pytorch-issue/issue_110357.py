import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        param = torch.rand(2, 2)
        self.register_parameter('param', nn.Parameter(param))

mymodule = MyModule()

print("Print #1")
for param in mymodule.parameters():
    print(param)
print("Print #2")
for param in mymodule.parameters():
    print(param)

mydict = dict(param=mymodule.parameters())

print("Print #3")
for param in mydict['param']:
    print(param)
print("Print #4")
for param in mydict['param']:
    print(param)