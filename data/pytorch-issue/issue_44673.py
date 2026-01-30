import torch

# directly calling c10d APIs from python
with torch.autograd.profiler.profile() as prof:
    pending_work = []
    for i in range(5):
        w = dist.all_reduce(torch.ones(i), async_op=True)
        pending_work.append(ww)
    for w in pending_work:
        w.wait()
            
# Should show time and ideally shapes for each input
print(prof.key_averages.table())

# DDP training, should include profiling of c10d calls. 
ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])
# forward pass of models with c10d collectives should be profiled. 
with torch.autograd.profiler.profile() as prof:
    outputs = ddp_model(torch.randn(20, 10).to(torch.cuda.current_device()))
    loss_fn(outputs, labels).backward()

with profiler():
    with record_funcion("allreduce"):
        work = dist.all_reduce(async_op = True)
    work.wait()