import torch

class CustomRepeatInterleave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, repeats):
        ctx.repeats = repeats
        output = input.repeat_interleave(repeats)
        ctx.mark_dirty(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        repeats = ctx.repeats
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        for i in range(repeats):
            grad_input += grad_output[i]  # Fixed the closing parenthesis here
        return grad_input, None

# Example usage
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
repeats = 2
y = CustomRepeatInterleave.apply(x, repeats)

z = y.sum()
z.backward()

print(x.grad)