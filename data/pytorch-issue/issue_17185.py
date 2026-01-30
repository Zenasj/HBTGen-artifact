import torch
a = torch.tensor([1,2,3]).view(1,3)
b = torch.tensor([4,5]).view(2,1)
print(a)
print(b)
print(a*b) # directly multiplying will trigger the correct broadcast

print(a.expand(2,3)*b.expand(2,3)) # explicitly using expand to broadcast followed by * is also fine

a *= b # trying to broadcast with *= doesn't work

a = a.expand(2,3) # explicitly broadcasting with expand
b = b.expand(2,3) # explicitly broadcasting with expand
a *= b # this produces weird values that I don't understand
print(a)

a = torch.rand(1,3)
b = torch.rand(2,1)
a *= b