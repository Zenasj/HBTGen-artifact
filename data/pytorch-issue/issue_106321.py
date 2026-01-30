import torch
import torch._dynamo


def foo(xs, state, w):
    outputs = []
    for x in xs:
        state = x + state * w
        outputs.append(state)
    return torch.cat(outputs)


device = torch.device("cuda")

xs = torch.rand((5, 3, 7), requires_grad=True, device=device, dtype=torch.float64)
state = torch.rand((3, 7), requires_grad=True, device=device, dtype=torch.float64)
w = torch.rand((7,), requires_grad=True, device=device, dtype=torch.float64)

c_foo = torch.compile(
    foo,
    mode="reduce-overhead",
)
torch._dynamo.reset()

torch.autograd.gradcheck(c_foo, (xs, state, w))  # GradcheckError