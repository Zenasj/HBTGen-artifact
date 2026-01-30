import torch

print('Part 1')
c = torch.tensor([0, 0.5], device='mps')
print(f'GOOD: {c=}')
print(f'GOOD: c[1] directly is {c[1]}')
print(f'BAD:  {c[1]=}, should be 0.5')
print(f'BAD:  {str(c[1])=}, should be 0.5')
print(f'BAD:  {repr(c[1])=}, should be 0.5')

print()
print('Part 2')
c = torch.tensor([1, 0.5], device='mps')
print(f'GOOD: {c=}')
print(f'GOOD: {c[1]=}')
print(f'GOOD: {str(c[1])=}')
print(f'GOOD: {repr(c[1])=}')

print(f'BAD:  {c[1][(Ellipsis, None)]=}, should be 0.5')
print(f'BAD:  {c[1]=}, should be 0.5')
print(f'GOOD: {c[1].clone()=}')
print(f'GOOD: {c[1].to("cpu").to("mps")=}')