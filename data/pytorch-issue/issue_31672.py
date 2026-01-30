import torch
print("My pytorch version {}\n\n".format(torch.__version__))

def test_index_put(device, dtype):
    print("===== {}, {} =====".format(str(device), str(dtype)))
    
    A = torch.zeros((2,2), device = device, dtype = dtype)
    values = torch.as_tensor([1, 1, 1], device = device,dtype = dtype)
    rows =  torch.as_tensor([0, 1, 1], device = device, dtype = torch.int64)
    cols =  torch.as_tensor([0, 0, 0], device = device, dtype = torch.int64)
    idxes = (rows, cols)
    A.index_put_(idxes, values, accumulate=True)

    print(A)

test_index_put('cpu', torch.float32)
test_index_put('cpu', torch.int64)
test_index_put('cuda', torch.float32)
test_index_put('cuda', torch.int64)