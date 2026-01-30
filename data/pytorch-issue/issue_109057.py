import torch

@torch.compile(backend="eager", fullgraph=True)
def f(t):
    batch_size_per_rank = [2, 2, 2, 2, 2, 2]
    if all(bs == batch_size_per_rank[0] for bs in batch_size_per_rank):
        return t
    else:
        return 1 - t
        
if __name__ == "__main__":
    torch.manual_seed(42)
    print(f(torch.randn(1)))