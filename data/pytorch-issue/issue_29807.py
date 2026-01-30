import torch
import torch.nn as nn
class MyModule(nn.Module):
    def __init__(self):

        super(MyModule, self).__init__()

        params = [torch.rand(1,3) for i in range(4)]
        self.params = []
        for i, param in enumerate(params):
            self.register_buffer('param_{}'.format(i), param)
            self.params.append(getattr(self, 'param_{}'.format(i)))

a = MyModule()
print(a.state_dict())
print(a.params)

a.cuda()
print(a.state_dict())
print(a.params)