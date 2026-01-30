import torch

@torch.library.custom_op("mylib::mycube", mutates_args=())
def mycube(x: Tensor) -> Tensor:
    return x ** 3, 3 * x ** 2

def setup_context(ctx, inputs, output) -> Tensor:
    ctx.save_for_backward(inputs[0], output[1])
    ctx.save_for_forward(inputs[0], output[1])

def backward(ctx, grad_output, grad_saved):
    input, dinput = ctx.saved_tensors
    return grad_output * dinput + 6 * grad_saved * input

def myjvp(ctx, input_tangent):
    print(3)
    input, dinput = ctx.saved_tensors
    return input_tangent * dinput, 6 * input_tangent * input

mycube.register_autograd(backward, setup_context=setup_context)

mycube.register_jvp(myjvp, setup_context=setup_context)

x = torch.tensor([1., 2., 3.], requires_grad=True)  # 
# y = mycube(x)
# grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
value, grad = torch.func.jvp(mycube, (x, ), (torch.ones(3), ))
print(value, grad)
# RuntimeError: Unable to cast (GradTrackingTensor(...) to Tensor