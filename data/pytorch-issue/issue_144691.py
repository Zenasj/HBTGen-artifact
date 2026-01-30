import torch.nn as nn

import torch


class Model(torch.nn.Module):
    def forward(
        self,
        input_ids,
        image_features,
        vocab_size,
    ):
        if image_features.numel():
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            # positions for image tokens
            condition = (input_ids < 0) & (input_ids > -int(1e9))
            positions = torch.where(condition)
            # has_image = len(positions[0].tolist()) > 0
            input_ids = input_ids.clamp_min(0).clamp_max(vocab_size)

            return (input_ids, *positions)

        return (input_ids, *torch.where(torch.zeros((1, 1), dtype=torch.bool)))


inputs = [
    (
        (torch.arange(24) - 8).reshape((2, -1)).to(torch.int64),
        torch.arange(32).reshape((2, -1)).to(torch.float32),
        1025,
    ),
    (
        (torch.arange(24) - 8).reshape((2, -1)).to(torch.int64),
        torch.tensor([[], []], dtype=torch.float32),
        1025,
    ),
]
model = Model()
expected = [model(*inp) for inp in inputs]
assert len(expected) == 2
assert len(expected[0]) == len(expected[1]) == 3


# Rewriting with torch.cond.

class Model2(torch.nn.Module):
    def forward(self, input_ids, image_features, vocab_size):
        def then_branch(input_ids, image_features, vocab_size):
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            condition = (input_ids < 0) & (input_ids > -int(1e9))
            positions = torch.nonzero(condition, as_tuple=True)
            input_ids = input_ids.clamp_min(0).clamp_max(vocab_size)
            return (input_ids, positions[0], positions[1])

        def else_branch(input_ids, image_features, vocab_size):
            r = torch.where(torch.zeros((1, 1), dtype=torch.bool))
            return (input_ids, r[0], r[1])

        a, b, c = torch.cond(
            image_features.numel() > 0,
            then_branch,
            else_branch,
            [input_ids, image_features, vocab_size],
        )
        return a, b, c

# Check that it is equivalent.
model2 = Model2()
new_out = [model2(*inp) for inp in inputs]
for i in range(2):
    for j in range(3):
        torch.testing.assert_close(expected[i][j], new_out[i][j])

batch = torch.export.Dim("batch")
seq_length = torch.export.Dim("seq_length")
dynamic_shapes = ({0: batch}, {0: batch, 1: seq_length}, None)

# We try to export with (tensor, tensor, int)
# ep = torch.export.export(model2, inputs[0], dynamic_shapes=dynamic_shapes, strict=False)
# fails with Expect operands to be a tuple of possibly nested dict/list/tuple that only consists of tensor leaves, but got [FakeTensor(..., size=(s1, 12), dtype=torch.int64), FakeTensor(..., size=(s2, s3)), 1025].
# print(ep)


# We try to export with (tensor, tensor, int)
new_inputs = (*inputs[0][:2], torch.tensor([1025], dtype=torch.int64))
# ep = torch.export.export(model2, new_inputs, dynamic_shapes=dynamic_shapes, strict=False)
# torch._dynamo.exc.Unsupported: dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.exc.UncapturedHigherOrderOpError: Cond doesn't work unless it is captured completely with torch.compile. Scroll up to find out what causes the graph break.
# print(ep)

torch._dynamo.config.capture_dynamic_output_shape_ops = True
ep = torch.export.export(model2, new_inputs, dynamic_shapes=dynamic_shapes, strict=False)
# torch._dynamo.exc.UncapturedHigherOrderOpError: Expected true_fn_output and false_fn_output to have same metadata but found:
# pair[1] differ in 'shape: torch.Size([u0]) vs torch.Size([u1])', where lhs is FakeTensor(..., size=(u0,), dtype=torch.int64) and rhs is FakeTensor(..., size=(u1,), dtype=torch.int64)
# pair[2] differ in 'shape: torch.Size([u0]) vs torch.Size([u1])', where lhs is FakeTensor(..., size=(u0,), dtype=torch.int64) and rhs is FakeTensor(..., size=(u1,), dtype=torch.int64)
print(ep)