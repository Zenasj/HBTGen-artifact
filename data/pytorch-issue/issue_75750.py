import torch

def callback():
    print("Callback! Raising an error...")
    raise RuntimeError("Error from callback!")

def hook_with_callback(*args):
    print("Backward hook!")
    torch.autograd.Variable._execution_engine.queue_callback(callback)

t = torch.tensor([1., 2.], requires_grad=True, device=torch.device("cuda"))
t.register_hook(hook_with_callback)
output = t ** 2
loss = output.sum()
loss.backward()