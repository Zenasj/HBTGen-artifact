import torch
x = torch.tensor([1], device='cuda:0', dtype=torch.float64)

def forward():
    a = torch.tensor([[0]], device='cuda:0', dtype=torch.float64)
    # Move it out of the function will eliminate the error
    
    a1 = a # This line and the next are required to reproduce the error
    a2 = a1
    if x[0] >= 0:
        a.transpose(0, 1) # This line is required for reproduction
        a2[0, 0] = 0 
    return (a1, ) # replace it with "return a1", and the error is eliminated. 

print(forward())
fn_compiled = torch.compile(forward)
print(fn_compiled())