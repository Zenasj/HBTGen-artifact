import torch

def test(batch_size):
    mat = torch.randn(batch_size, 12, 12)
    vec = torch.randn(batch_size, 12, 1)

    res, _ = torch.solve(vec, mat)
    print("CPU res: {}".format(torch.norm(torch.bmm(mat,res) - vec)))

    mat = mat.cuda()
    vec = vec.cuda()

    res, _ = torch.solve(vec, mat)
    res, _ = torch.solve(vec, mat)
    print("GPU res: {}".format(torch.norm(torch.bmm(mat,res) - vec)))

test(batch_size=65535)
print()
test(batch_size=65536)

import torch

def test(batch_size):
    mat = torch.randn(batch_size, 12, 12)
    vec = torch.randn(batch_size, 12, 1)

    res, _ = torch.solve(vec, mat)
    print("CPU res: {}".format(torch.norm(torch.bmm(mat,res) - vec)))

    mat = mat.cuda()
    vec = vec.cuda()

    res, _ = torch.solve(vec, mat)
    try:
        res, _ = torch.solve(vec, mat)
    except:
        pass
    print("GPU res: {}".format(torch.norm(torch.bmm(mat,res) - vec)))

test(batch_size=65535)
print()
test(batch_size=65536)

def GPU_solve(As, bs):
    batch_size, N, N = As.shape
    Ws = As.new(size=(*As.shape[:2], 1))
    smb = 65535
    temp_LR = As.new(size=(smb, *As.shape[1:]))
    for i in range(batch_size // smb + 1):
        start = smb * i
        end = smb * (i + 1) if i < batch_size // smb else batch_size
        torch.solve(bs[start:end], As[start:end], out=(Ws[start:end], temp_LR[0:end-start]))
    return Ws