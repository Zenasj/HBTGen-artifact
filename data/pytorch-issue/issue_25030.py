import torch
torch.manual_seed(0)
w = torch.tensor( [1.2899e-01, 6.2532e-01, 3.6483e-02, 1.5196e-01, 2.9675e-03, 
4.9773e-03,4.5881e-02, 2.9019e-03, 5.2139e-04, 1.5281e-17] )
print(torch.multinomial(w, 10, replacement=False))