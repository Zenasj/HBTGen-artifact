import torch.nn as nn

import numpy as np
import torch
from torch import nn

# Very basic module: packs a sequence and then
# unpacks it:

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
                
    def forward(self, x, lengths):
        packed_seq = nn.utils.rnn.pack_padded_sequence(x,
                                                       lengths,
                                                       batch_first=True,
                                                       enforce_sorted=True)

        padded_seq = nn.utils.rnn.pad_packed_sequence(packed_seq,
                                                      batch_first=True,
                                                      padding_value=0.0)
        return padded_seq[0].double(), padded_seq[1].long()

# Instantiate module:
module_instance = Net()

# Generate inputs and see outputs:
x = torch.rand(4, 10, 3).double()
lengths = torch.tensor([10, 8, 4, 2]).long()

result_no_onnx = module_instance(x, lengths)

# Export to onnx:
torch.onnx.export(model=module_instance, 
                  args=(x, lengths), 
                  f="packer.onnx", 
                  example_outputs=(result_no_onnx[0], result_no_onnx[1]),
                  verbose=True, 
                  input_names=["in", "lengths"], 
                  output_names=["out", "lengths_out"],
                  dynamic_axes={"in": {0: "batch",
                                       1: "seq"},
                                "lengths": {0: "batch"},
                                "out": {0: "batch",
                                        1: "seq"},
                                "lengths_out": {0: "batch"}}
                 )

# Load the exported model into onnxruntime and
# run the module:
import onnxruntime as rt

sess = rt.InferenceSession("packer.onnx")
result_onnx = sess.run(["out", "lengths_out"], 
                       input_feed={"in": x.numpy().astype(np.float64),
                                   "lengths": lengths.numpy().astype(np.int64)
                                  }
                      )

# Compare the outputs generated with and without onnx:
assert np.all(result_no_onnx[0].numpy() == result_onnx[0])

# And the assertion fails, as the exported ONNX did not pad
# with zeros the packed sequence.