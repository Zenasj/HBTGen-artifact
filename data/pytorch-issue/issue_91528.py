import torch.nn as nn

import torch
import onnx
from typing import Tuple
from torch import nn, Tensor
import numpy as np
import onnxruntime
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

@torch.jit.script
def sort_tensor(context: Tensor, lens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    lens_sorted, ids_sorted = torch.sort(lens, descending=True)
    unsort_ids = torch.zeros_like(ids_sorted)
    for i in range(ids_sorted.shape[0]):
        unsort_ids[ids_sorted[i]] = i
    context = context[ids_sorted]
    return context, lens_sorted, unsort_ids

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, max_batch_size=64):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Create a buffer of zeroes to be passed as hx parameter to LSTM()                                                                                                                    
        self.real_hidden_size: int = self.bilstm.proj_size if self.bilstm.proj_size > 0 else self.bilstm.hidden_size
        self.n_dir: int = int(self.bilstm.num_layers * 2)
        h_zeros = torch.zeros(self.n_dir * max_batch_size * max(self.real_hidden_size, self.bilstm.hidden_size))
        self.register_buffer('h_zeros', h_zeros, persistent=False)
        self.bilstm.flatten_parameters()

    def lstm_tensor(self, context: Tensor, lens: Tensor, enforce_sorted: bool = False) -> Tuple[Tensor, Tensor]:
        seq = nn.utils.rnn.pack_padded_sequence(
            context, lens.long().cpu(), batch_first=True, enforce_sorted=enforce_sorted
        )
        return self.lstm_sequence(seq)

    def lstm_sequence(self, seq: PackedSequence) -> Tuple[Tensor, Tensor]:
        if not (torch.jit.is_scripting() or torch.jit.is_tracing()):
            self.bilstm.flatten_parameters()
            ret, _ = self.bilstm(seq)
        else:
            # Calculate sizes and prepare views to our zero buffer to pass as hx                                                                                                              
            max_batch_size = seq.batch_sizes[0]
            common_shape = (self.n_dir, max_batch_size)
            # borisf: ONNX inssists on having self.n_dir float 
            # WAR: common_size = max_batch_size.float().mul(self.n_dir).long()                                                                                                                
            common_size = max_batch_size * self.n_dir
            h_shape = (*common_shape, self.real_hidden_size)
            c_shape = (*common_shape, self.bilstm.hidden_size)
            hx = (
                self.h_zeros[: common_size.mul(self.real_hidden_size)].view(h_shape),
                self.h_zeros[: common_size.mul(self.bilstm.hidden_size)].view(c_shape),
            )
            ret, _ = self.bilstm(seq, hx)
        return nn.utils.rnn.pad_packed_sequence(ret, batch_first=True)

    def forward(self, context: Tensor, lens: Tensor) -> Tensor:
        context, lens, unsort_ids = sort_tensor(context, lens)
        dtype = context.dtype
        return self.lstm_tensor(context, lens, enforce_sorted=True)[0][unsort_ids]

def get_mask_from_lengths_and_val(lengths, val):
    max_len = val.shape[-1]
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask

class ConvLSTMLinear(nn.Module):
    def __init__(self,n_channels=256):
        super(ConvLSTMLinear, self).__init__()
        self.bilstm = BiLSTM(n_channels, int(n_channels // 2), 1)
        self.embedding = torch.nn.Embedding(185, 256)

    def forward(self, context: Tensor, lens: Tensor) -> Tensor:
        context = self.embedding(context).transpose(1, 2)
        mask = get_mask_from_lengths_and_val(lens, context).unsqueeze(1)
        # borisfom: works if you return this, but fails with LSTM.                                                                                                      
        # return context * mask                                                                                                                                                               
        # return sort_tensor(context, lens)[0]                                                                                                                                                
        return self.bilstm((context * mask).transpose(1,2), lens)

def get_sample(max_batch, max_dim, device='cuda'):
    inp = torch.randint(16, 32, (max_batch, max_dim), device=device, dtype=torch.int64)
    lens = torch.randint(max_dim // 8, max_dim // 4, (max_batch,), device=device, dtype=torch.int64)
    return (inp, lens)
with torch.no_grad(), torch.inference_mode():
    model = ConvLSTMLinear().cuda().eval()
    model2 = ConvLSTMLinear().cuda().eval()

    input = get_sample(1, 333)
    input2 = get_sample(7, 555)

    output = model(*input2)
    output2 = model2(*input2)
    model  = torch.jit.trace_module(model,
                                   { 'forward': input},
                                   strict = True,
                                   check_trace = [input2])

    print ("Running traced model")
    output1 = model(*input2)
    print("Traced model comparison: ", output1.sum()-output.sum())

    print ("ONNX export")
    # export with `dynamic_axes`                                                                                                                                                              
    torch.onnx.export(model2, input, 'lstm.onnx',
                      input_names=['input', 'lens', ],
                      output_names=['output',],
                      dynamic_axes={'input': {0: 'batch', 1: 'sequence'},
                                    'output': {0: 'batch', 1: 'sequence'},
                                    'lens': {0: 'batch'},
                                    },
                      opset_version=17,
                      verbose=True
    )
    onnx_model = onnx.load('lstm.onnx')
    onnx.checker.check_model(onnx_model)
    print ("ONNX export is done")

    onnx_session_opt = onnxruntime.SessionOptions()
    onnx_session_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), sess_options=onnx_session_opt, providers=['CUDAExecutionProvider']
    )

x_onnx = sess.run(None, {'input': input2[0].cpu().numpy(), 'lens' : input2[1].cpu().numpy()})[0]

print(output.shape, x_onnx.shape)
np.testing.assert_almost_equal(x_onnx, output2.cpu().detach(), decimal=3)
print("All comparisons passed")