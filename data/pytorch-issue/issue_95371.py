import torch
input=torch.sparse_coo_tensor([[]], [], (3,), device='cuda')
def test():
    result=torch.sparse.log_softmax(input,0)
    return result
print(test())