import torch

def describe(x):
    return f"dtype: {x.dtype}, shape: {tuple(x.shape)}, stride: {x.stride()}, contiguous: {x.is_contiguous()}, storage offset: {x.storage_offset()}, device: {x.device}"

def check_take_along_dim(x, index):
    a = torch.take_along_dim(x, index, dim=0)
    b = torch.take_along_dim(x, index.contiguous(), dim=0)
    return torch.equal(a, b)
    
def check_gather(x, index):
    a = torch.gather(x, dim=0, index=index)
    b = torch.gather(x, dim=0, index=index.contiguous())
    return torch.equal(a, b)

def contiguous_test(device="cpu"):
    x = torch.randn(10, 100, 4).to(torch.float64).to(device)
    print("x =", describe(x))

    i = torch.randint(10, (100,)).to(device)

    j = i.unsqueeze(0).unsqueeze(2).expand(-1, -1, x.shape[2])
    print("j =", describe(j))

    # always OK
    print("gather j:", "OK" if check_gather(x, j) else "!!KO!!")
    print("take along dim j:", "OK" if check_take_along_dim(x, j) else "!!KO!!")

    k = i.unsqueeze(1).t().unsqueeze(2).expand(-1, -1, x.shape[2])
    print("k =", describe(k))

    # KO on cpu, OK on cuda
    print("gather k:", "OK" if check_gather(x, k) else "!!KO!!")
    print("take along dim k:", "OK" if check_take_along_dim(x, k) else "!!KO!!")