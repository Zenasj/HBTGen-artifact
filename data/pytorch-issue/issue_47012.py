import torch

# implicit SPMD mode: when device_ids is missing, DDP will use all visible devices
ddp_model = DistributedDataParallel(model)

# explicit SPMD mode: by specifying devices
ddp_model = DistributedDataParallel(model, device_ids=[d1, d2, d3, d4])

#SPSD mode
ddp_model = DistributedDataParallel(model, device_ids=[d1])

with dist_autograd.context() as ctx_id:
	fut_outs = []
	for ddp in ddp_modules:
		fut_outs.append(ddp.rpc_async().forward(inps))

	gathered_outs = torch.concat(torch.futures.wait_all(fut_outs))
	dist.backward(ctx_id, [loss_fn(gathered_outs)])