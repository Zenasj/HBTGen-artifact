import torch

print("CPU (the following medians should be equal and in fact are)")
x = torch.randn(3, 3)
median_of_diag1 = x.diagonal(dim1=-1, dim2=-2).median(-1)[0]
print("median_of_diag1", median_of_diag1)
median_of_diag2 = x.diagonal(dim1=-1, dim2=-2).clone().median(-1)[0]
print("median_of_diag2", median_of_diag2)

print("GPU (the following medians should be equal but are not)")
x = x.cuda()
median_of_diag1 = x.diagonal(dim1=-1, dim2=-2).median(-1)[0]
print("median_of_diag1", median_of_diag1)
median_of_diag2 = x.diagonal(dim1=-1, dim2=-2).clone().median(-1)[0]
print("median_of_diag2", median_of_diag2)

In [20]: a=torch.arange(9, device="cuda").resize_(3,3)                                                                                                                                                   

In [21]: a[:,0].median(-1)[0]                                                                                                                                                                            
Out[21]: tensor(1, device='cuda:0') 

In [22]: a[:,0]      # 1 is not even part of this slice                                                                                                                                                                                    
Out[22]: tensor([0, 3, 6], device='cuda:0')