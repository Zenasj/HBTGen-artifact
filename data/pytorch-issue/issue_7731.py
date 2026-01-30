import torch

## Addition
x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

t1 += t2
print("Incorrect answer")
print(t1)

x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

print("Correct answer")
print(t1 + t2)

## Subtraction
x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

t1 -= t2
print("Incorrect answer")
print(t1)

x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

print("Correct answer")
print(t1 - t2)

## Multiplication
x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

t1 *= t2
print("Incorrect answer")
print(t1)

x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

print("Correct answer")
print(t1 * t2)

## Division
x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

t1 /= t2
print("Incorrect answer")
print(t1)

x = torch.Tensor([[1],[2],[3]])
t1 = x.expand(3, 3)
t2 = x.t().expand(3, 3)

print("Correct answer")
print(t1 / t2)