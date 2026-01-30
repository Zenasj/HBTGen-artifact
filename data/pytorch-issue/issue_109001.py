import torch
from torch.testing._internal.common_methods_invocations import python_ref_db, op_db

device = "cuda:0"
ops = [('_refs.isreal', torch.float32), ('_refs.fft.ihfftn', torch.float32)]

for (op_name, dtype) in ops:
  op = None
  for found_op in python_ref_db + op_db:
    if found_op.name == op_name:
      op = found_op
      break

  assert op is not None

  samples = found_op.sample_inputs(device, dtype)

  for sample in samples:
    @torch.compile
    def fn(sample):
      expected = op(sample.input, *sample.args, **sample.kwargs)
      out = torch.zeros_like(expected)
      init_strides = out.stride()

      op(sample.input, out=out, *sample.args, **sample.kwargs)

      final_strides = out.stride()
      if init_strides != final_strides:
        print("FAILED", op_name, expected, out, init_strides, final_strides)
    
    fn(sample)

x = torch.tensor([-8.4784-1.7658j])
y = torch.tensor([-8.4784-1.7658j])
ans = torch.compile(torch.matmul)(x, y)
out = torch.empty_like(ans)
torch.compile(torch.matmul)(x, y, out=out)
torch.testing.assert_close(ans, out) # fails