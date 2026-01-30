import torch

ts = torch.load('list_of_tensors.pt', weights_only=True)
# ts = ts[1::2]  # this is used to get a smaller test case
norms = torch.stack([torch.norm(t) for t in ts])
foreach_norms = torch.stack(torch._foreach_norm(ts))
if not torch.allclose(norms, foreach_norms):
	print(f"norms differ:\n{norms}\n{foreach_norms}")
	# torch.save(ts, 'list_of_tensors.pt')  # this is used to get a smaller test case

[torch.Size([4915200]), torch.Size([4915200]), torch.Size([4915200]), torch.Size([2457600]), torch.Size([0]), torch.Size([0]), torch.Size([4915200])]