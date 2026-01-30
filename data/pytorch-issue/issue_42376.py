import torch

def make_lr_scheduler():
    param1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    sgd = torch.optim.SGD([param1], lr=0.1, momentum=0.9)
    return torch.optim.lr_scheduler.OneCycleLR(sgd, 6.0, total_steps=10)

lr_scheduler = make_lr_scheduler()
print(lr_scheduler.anneal_func)

with open("checkpoint", "wb") as f:
    torch.save(lr_scheduler.state_dict(), f)
    
lr_scheduler2 = make_lr_scheduler()
print(lr_scheduler2.anneal_func)

with open("checkpoint", "rb") as f:
    lr_scheduler2.load_state_dict(torch.load(f))
print(lr_scheduler2.anneal_func)