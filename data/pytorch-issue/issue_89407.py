import torch

@torch.inference_mode() # required
def cake():
    mps = torch.device("mps")
    a = torch.arange(5, device = mps)
    test_idx = 1 # 1..*
    assert torch.equal(
        a[test_idx].unsqueeze(0),
        a[test_idx].cpu().unsqueeze(0).to(mps)
    )
cake()