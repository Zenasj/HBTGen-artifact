import torch

if __name__ == '__main__':
    bmm_weight = torch.ones((8, 8), dtype=torch.float32, requires_grad=True)
    a = torch.ones((5, 8, 8), dtype=torch.float32) * torch.arange(5)[:, None, None]

    b = bmm_weight * a

    b = b.sum((1, 2))
    grads = torch.autograd.grad(b, bmm_weight, torch.eye(len(b), dtype=b.dtype, device=b.device),
                                retain_graph=True, create_graph=True, is_grads_batched=True)

    print(grads[0].shape)