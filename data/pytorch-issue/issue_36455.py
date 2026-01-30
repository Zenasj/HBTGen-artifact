import torch

tensor = torch.ones((2,))
new_tensor = tensor.new_full((3, 4), 3.141592, requires_grad=True)
print(new_tensor.requires_grad)
# False
tensor = torch.ones((2,), requires_grad=True)
new_tensor = tensor.new_full((3, 4), 3.141592, requires_grad=True)
print(new_tensor.requires_grad)
# False
new_tensor.requires_grad = True
print(new_tensor.requires_grad)
# True

wrap(dispatch_new_zeros/full/empty(self, _r.intlist(0), options));

THPVariable_Wrap()