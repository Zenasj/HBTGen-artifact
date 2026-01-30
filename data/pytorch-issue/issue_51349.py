import torch
torch.autograd.set_detect_anomaly(True) # if comment out this, no leak
for rep in range(1000):
    batch = torch.ones(10, 30000).cuda().requires_grad_()
    y=batch.norm()
    grad = torch.autograd.grad(
        outputs=y, inputs=batch,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    print("memory allocated:", float(torch.cuda.memory_allocated()/1024**2), "MiB")
    pass