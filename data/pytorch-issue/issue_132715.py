import torch

device = "cuda"
dtype = torch.bfloat16

with torch.autocast(device_type=device, enabled=True, dtype=dtype):
    t = torch.randn([3, 4, 5], dtype=dtype, device=device, requires_grad=True)
    index = torch.randint(
        low=0, high=3, size=[3, 4, 5], dtype=torch.int64, device=device
    )
    val = torch.randn(1, dtype=dtype, device=device)

    res = torch.index_put(t, [index], val)

    loss = res.mean()
    loss.backward()
    print(t.grad)