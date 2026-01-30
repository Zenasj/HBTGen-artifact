import torch
torch.__version__
print(f'torch.__version__={torch.__version__}')
data1 = torch.rand(5, 5)
data1[0] = float('nan')
data2 = data1
result = torch.equal(data1, data2)
print(f'result={result}')