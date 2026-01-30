import torch

logits = torch.randn((1, 4, 3)) # Failed

def test_fn2(primals_1, primals_2):
    full_default = torch.ops.aten.full.default([1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index = torch.ops.aten.index.Tensor(primals_1, [full_default, primals_2]);  primals_1 = None
    return index
sequence_lengths = torch.tensor([-1])
correct3 = test_fn2(logits, sequence_lengths)
compiled3 = torch._dynamo.optimize_assert(compile_fx)(test_fn2)(logits, sequence_lengths)
print("correct3 is: {}".format(correct3), flush=True)
print("compiled3 is: {}".format(compiled3), flush=True)