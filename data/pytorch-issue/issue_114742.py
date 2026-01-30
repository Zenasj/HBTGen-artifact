import torch

def model_add(params):
    result = torch.add(**params)

    return result

ifm = torch.randint(low=-50,high=50,size=(8, 24, 24, 3),dtype = torch.int32)
other = int(0.9)
alpha = int(0.4)

out = torch.empty([1],dtype=torch.float32)

params = {"other": other, "alpha":alpha, "input": ifm}
foo = torch.compile(model_add, backend="eager")
output = foo(params)
print(output)