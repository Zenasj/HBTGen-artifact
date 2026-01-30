import torch
import torch._dynamo
import torch._dynamo.config
def mySum64(x):
    return (x+x).to(torch.int64)

x = torch.tensor( (2147483647), dtype=torch.int32)
torchResult = mySum64(x)
dynamoResult = torch.compile(mySum64)(x)

print(torchResult)
print(dynamoResult)

tensor(-2)
tensor(4294967294)