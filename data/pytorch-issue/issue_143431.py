import torch

def f(image_latent):
    B = 2
    num_ref = 3
    num_tar = 3
    x = torch.rand(B, 12)
    indices = torch.argsort(torch.rand(*x.shape), dim=-1)[:, :num_ref + num_tar]
    return image_latent[torch.arange(B).unsqueeze(-1), indices][:, :num_ref]

torch.manual_seed(54321)
torch.cuda.manual_seed_all(54321)
print(torch.compile(backend="aot_eager", fullgraph=True)(f)(torch.randn((2, 12, 16, 32, 32), device='cuda')).sum())

torch.manual_seed(54321)
torch.cuda.manual_seed_all(54321)
print(torch.compile(backend="aot_eager", fullgraph=True)(f)(torch.randn((2, 12, 16, 32, 32), device='cuda')).sum())

torch.manual_seed(54321)
torch.cuda.manual_seed_all(54321)
print(torch.compile(backend="eager", fullgraph=True)(f)(torch.randn((2, 12, 16, 32, 32), device='cuda')).sum())

torch.manual_seed(54321)
torch.cuda.manual_seed_all(54321)
print(torch.compile(backend="eager", fullgraph=True)(f)(torch.randn((2, 12, 16, 32, 32), device='cuda')).sum())

tensor(209.5920, device='cuda:0')
tensor(209.5920, device='cuda:0')
tensor(300.4904, device='cuda:0')
tensor(300.4904, device='cuda:0')