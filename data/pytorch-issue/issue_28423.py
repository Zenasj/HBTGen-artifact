import torch.nn as nn

import torch
from torch import nn

class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

    def forward(self, x, x_lengths):
        x_packed = nn.utils.rnn.pack_padded_sequence(x, 
                                                     lengths=x_lengths, 
                                                     batch_first=False,
                                                     enforce_sorted=True)
        x_unpacked, out_seq_length = nn.utils.rnn.pad_packed_sequence(x_packed, 
                                                                      batch_first=False, 
                                                                      padding_value=0.0)
        return x_unpacked

# Prepare inputs:
example_input = (torch.Tensor([[[1,2], [3,4], [5,6]], [[1,2], [3,4], [0,0]]])
                 .permute(1,0,2)
                 )
print("Example input: %s" % example_input)
print("Example input shape: %s" % list(example_input.size()))

example_input_length = torch.Tensor([3,2]).long()

# Model instantiation and scripting
my_net = DummyNet()
my_net_script = torch.jit.script(my_net)

# Check that forward pass in scripted module works as expected:
output = my_net_script(example_input, example_input_length)
print("Output from TorchScript Module (works): %s" % output)

# Export to ONNX (fails):
torch.onnx.export(model=my_net_script,
                  args=(example_input, example_input_length),
                  example_outputs=output,
                  f="test_pack_unpack.onnx",
                  verbose=True,
                  input_names=["x", "x_length"],
                  output_names=["preds"]
                 )

import numpy as np

import torch
from torch import nn

class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

    def forward(self, x, x_lengths):
        x_packed = nn.utils.rnn.pack_padded_sequence(x, 
                                                     lengths=x_lengths, 
                                                     batch_first=False,
                                                     enforce_sorted=True)
        x_unpacked, out_seq_length = nn.utils.rnn.pad_packed_sequence(x_packed, 
                                                                      batch_first=False, 
                                                                      padding_value=0.0)
        return x_unpacked, out_seq_length

# Prepare inputs:
example_input = (torch.Tensor([[[1,2], [3,4], [5,6]], [[1,2], [3,4], [99,99]]])
                 .permute(1,0,2)
                 )
print("Example input:\n%s" % example_input)
print("Example input shape:\n%s" % list(example_input.size()))

example_input_length = torch.Tensor([3,2]).long()

# Model instantiation and scripting
my_net = DummyNet()
my_net_script = torch.jit.script(my_net)

# Check that forward pass in scripted module works as expected:
output, output_lengths = my_net_script(example_input, example_input_length)
print("Output from TorchScript Module (works):\n%s" % output, output_lengths)

# Export to ONNX (works, but the exported graph does almost nothing):
torch.onnx.export(model=my_net_script,
                  args=(example_input, example_input_length),
                  example_outputs=(output, output_lengths),
                  f="test_pack_unpack.onnx",
                  verbose=True,
                  input_names=["x", "x_length"],
                  output_names=["preds", "lengths"]
                 )

# Try inference with onnxruntime (could also be Caffe2, same results):
import onnxruntime as rt

sess = rt.InferenceSession("test_pack_unpack.onnx")
result_onnx = sess.run(["preds", "lengths"], 
                       input_feed={"x": example_input.numpy(),
                                   "x_length": example_input_length.numpy()
                                  }
                      )

output_onnx = result_onnx[0]
output_onnx_length = result_onnx[1]

# Check onnxrt output:
print("Output from onnxruntime (wrong):\n%s" % output_onnx, output_onnx_length)

# Assert that onnxruntime did nothing:
assert np.all(output.numpy() == output_onnx)

# The assertion fails (can be seen in the last print)