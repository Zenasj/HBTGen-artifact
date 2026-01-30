import torch

t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
s = t.shape[1]

t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
s = t.shape[1]

input = []
input.append(torch.tensor([1.0, 2.0, 3.0, 4.0]))
input.append(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
input.append(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
input[0].shape[0]
input[1].shape[1]
input[2].shape[2]

from subprocess import PIPE, Popen
process = Popen(['mypy', '33953.py'], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()
print(stdout)

b'33953.py:8: error: Tuple index out of range\n33953.py:9: error: Tuple index out of range\nFound 2 errors in 1 file (checked 1 source file)\n'