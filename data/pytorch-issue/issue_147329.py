import torch
cf = torch.compile(torch.isin)
elements = torch.tensor([1,2,3,4])
test_elements = 1
cf(elements,test_elements)