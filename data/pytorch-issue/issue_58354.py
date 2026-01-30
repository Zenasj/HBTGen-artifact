import torch

def install_hook(tensor):
    handle = None
    def hook(tensor):
        handle.remove()
        return torch.zeros_like(tensor)
    handle = tensor.register_hook(hook)

def test_hook():
    t = torch.ones((1, 5))
    t.requires_grad_()
    install_hook(t)
    (t ** 2).mean().backward()
    print(t.grad)

if __name__ == '__main__':
    test_hook()