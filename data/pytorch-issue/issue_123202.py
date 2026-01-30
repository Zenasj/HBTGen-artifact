import torch

@torch.compile(backend="eager")
def f():
    for _ in range(4):
        i = torch.randn(2, 3) # torch.tensor([[400], [500]], dtype=torch.float32)
        j = torch.randn(2, 3) # torch.tensor([[200], [300]], dtype=torch.float32)
        def c():
            print("graph break")
            loss = (i + j).sum()
            return loss
        init_val = c().item()
        def s():
            print("step")
            i.sub_(1)
            j.sub_(1)
        for _ in range(3):
            c()
            s()
        assert c().item() < init_val

    f()