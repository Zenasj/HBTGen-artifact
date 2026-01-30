import torch

print("good", torch.randn(5,5,5).max(1))
print("terrible", torch.randn(5,5,10).max(1))
print("not as good", torch.randn(5,5,500).max(1))
print ("old behaviour = gold standard")
print(tuple(torch.randn(5,5,5).max(1)))
print(tuple(torch.randn(5,5,10).max(1)))
print(tuple(torch.randn(5,5,500).max(1)))