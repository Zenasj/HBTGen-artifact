import torch

torch._logging.set_logs(recompiles=True)

def fn_simple(x):
    if torch.is_inference_mode_enabled():
        return x.sum()
    else:
        return x.min()

fn = torch.compile(fn_simple, fullgraph=True, backend="inductor")
x = torch.tensor([1, 2, 3], dtype=torch.float32)

fn(x)

with torch.inference_mode():
    x_inference = torch.tensor([1, 2, 3], dtype=torch.float32)
    fn(x_inference)

with torch.no_grad():
    x_no_grad = torch.tensor([1, 2, 3], dtype=torch.float32)
    fn(x_no_grad)