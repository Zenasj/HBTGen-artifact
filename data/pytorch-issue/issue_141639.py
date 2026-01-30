import torch
import torch.nn as nn


class ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, bias=None, compiled=True):
        CHUNK_SIZE = 1024  # Reduced for gradcheck

        def compute_loss(input_chunk, weight, bias, target):
            logits = torch.addmm(bias, input_chunk, weight.t())
            logits = logits.float()
            loss = torch.nn.functional.cross_entropy(logits, target)
            return loss

        grad_weight = torch.zeros_like(weight)
        grad_inputs = []
        grad_bias = torch.zeros_like(bias)
        loss_acc = torch.zeros((), device=_input.device)
        chunks = _input.shape[0] // CHUNK_SIZE

        def accumulate_chunk(input_chunk, target_chunk):
            (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), chunk_loss = (
                torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1, 2)
                )(input_chunk, weight, bias, target_chunk)
            )
            grad_weight.add_(chunk_grad_weight)
            grad_bias.add_(chunk_grad_bias)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        target_chunks = torch.chunk(target, chunks=chunks, dim=0)

        for input_chunk, target_chunk in zip(input_chunks, target_chunks):
            chunk_grad_input = accumulate_chunk(input_chunk, target_chunk)
            grad_inputs.append(chunk_grad_input)

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0) / chunks,
            grad_weight / chunks,
            grad_bias / chunks,
        )
        return loss_acc / chunks

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        return (grad_input, grad_weight, None, grad_bias, None)


torch.set_default_device("cuda")
torch.set_float32_matmul_precision("medium")

chunked_cross_entropy = ChunkedCE.apply
compiled_chunked_cross_entropy = torch.compile(chunked_cross_entropy)

B, T, D, V = 16, 128, 384, 267_735
model = nn.Linear(D, V, device="cuda")
x = torch.randn(B * T, D, requires_grad=True, device="cuda")
weight = torch.randn(V, D, requires_grad=True, device="cuda")
bias = torch.randn(V, requires_grad=True, device="cuda")
label = torch.randint(0, V, (B * T,), device="cuda")

compiled_x = x.detach().clone().requires_grad_(True)
compiled_weight = weight.detach().clone().requires_grad_(True)
compiled_bias = bias.detach().clone().requires_grad_(True)

out_compiled = compiled_chunked_cross_entropy(
    compiled_x.view(-1, D), compiled_weight, label.view(-1), compiled_bias
)
out_compiled.backward()

out = chunked_cross_entropy(
    x.view(-1, D), weight, label.view(-1), bias
)
out.backward()

print(x.grad)
print(compiled_x.grad)