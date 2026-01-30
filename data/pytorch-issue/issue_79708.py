import torch

param = torch.tensor([1., 2., 3.], requires_grad=True)

non_leaf_param = param.clone()

for i in range(2):
    inner_clone = non_leaf_param.clone()
    non_leaf_param.grad = None
    g, = torch.autograd.grad(inner_clone.sum(), non_leaf_param, create_graph=True)
    assert g is not None, f"grad is not set during iteration {i} for grad call"
    print('grad with grad:', g)
    inner_clone.sum().backward(inputs=[non_leaf_param], create_graph=True)
    assert non_leaf_param.grad is not None, f"grad is not set during iteration {i} for backward call"
    print('grad with backward:', non_leaf_param.grad)

    # In-place op messes up grad population during the second iteration.
    non_leaf_param.add_(1., alpha=0.1)

import torch

# `backward` in terms of `grad` that allows input= argument to work properly with in-place
# NB: you might run into warnings about non-leaf .grad being accessed
def hacky_backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None):
    if inputs is None:
        torch.autograd.backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        return
    grads = torch.autograd.grad(tensors, inputs, grad_tensors, retain_graph, create_graph)

    # Accumulate grad
    for grad, t in zip(grads, inputs):
        if t.grad is None:
            if grad.is_sparse:
                raise NotImplementedError
            # This replicates what retains_grad does, but it is not what accumulate grad
            # in real backward does (try to match the strides of the param in some cases)
            t.grad = grad.clone(memory_format=torch.contiguous_format)
        else:
            t.grad = t.grad + grad

backward = hacky_backward

param = torch.tensor([1., 2., 3.], requires_grad=True)
non_leaf_param = param.clone()

for i in range(2):
    print(f"[Iteration: {i}]")
    inner_clone = non_leaf_param.clone()
    non_leaf_param.grad = None

    backward(inner_clone.sum(), inputs=[non_leaf_param], create_graph=True)

    assert non_leaf_param.grad is not None, f"grad is not set during iteration {i} for backward call"

    # In-place op messes up grad population during the second iteration.
    non_leaf_param.add_(1., alpha=0.1)