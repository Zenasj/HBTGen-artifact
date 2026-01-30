import torch
device='cuda'
seq, d = 512, 128 
q = torch.rand((seq,seq),dtype=torch.bfloat16, device="cpu").to(device)
k = torch.rand((seq,seq), dtype=torch.bfloat16, device="cpu").to(device)
c = torch.zeros((seq,seq), dtype=torch.float32, device="cpu").to(device)

for op in [torch.add, torch.mul, torch.matmul]:
    try:
        op(q,k, out=c)
    except Exception as e:
        print(op.__name__)
        raise e