import torch.nn as nn

class MyModel:
    ...

class MyModel2:
    ...

ddp = DDP(MyModel())
local = MyModel2()
out = ddp(inp)
out = local(out)
out.backward()

class MyModel:
    ...

ddp = DDP(MyModel)
loss = ddp(data)
loss = loss / 10
loss.backward()

outputs = ddp(data)
loss = nn.CrossEntropyLoss(outputs, targets)
loss.backward() # starts backprop from outside of DDP