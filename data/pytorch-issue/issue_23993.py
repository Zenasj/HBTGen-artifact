import torch.nn as nn

class UnaryWrapper(nn.Module):
    def __init__(self, operation):
        super(UnaryWrapper, self).__init__()
        self.operation = operation

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        return self.operation(x)


class BinaryWrapper(nn.Module):
    def __init__(self, operation):
        super(BinaryWrapper, self).__init__()
        self.operation = operation

    def forward(self, x,y):
        return self.operation(x,y)


class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        self.mycat = UnaryWrapper(torch.cat)
        self.myadd= BinaryWrapper(torch.add)
        self.mymul = BinaryWrapper(torch.mul)
    
    def forward(self,x):
        y = self.mycat([x ,x ,x])
        z = self.myadd(y, y)
        z = self.mymul(z,z)
        return z

myModel = testModel()
x = torch.rand(3,5)
traced_model = torch.jit.trace(myModel,(x))
print(traced_model.code)