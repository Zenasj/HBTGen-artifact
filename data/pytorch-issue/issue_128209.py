import torch
import torch.nn as nn
import numpy as np

# Defining the model and putting it in eval mode:
tf = torch.nn.Transformer(
    d_model=64,
    nhead=8,
    num_encoder_layers=1,
    num_decoder_layers=2,
    dim_feedforward=256,
    batch_first=True
)
tf.eval()


# Creating dummy input data:
batch_size = 8
src_inp = []
src_pmask = []
tgt_inp = []
tgt_pmask = []

for idx in range(batch_size):
    seq_len_src = torch.randint(5, 10, (1,))[0]
    src_inp.append(torch.rand((seq_len_src, 64)))
    src_pmask.append(torch.zeros((seq_len_src)))

    seq_len_tgt = torch.randint(5, 10, (1,))[0]
    tgt_inp.append(torch.rand((seq_len_tgt, 64)))
    tgt_pmask.append(torch.zeros((seq_len_tgt)))

src_inp = torch.nn.utils.rnn.pad_sequence(src_inp, True, -1)
src_pmask = torch.nn.utils.rnn.pad_sequence(src_pmask, True, float("-inf"))
tgt_inp = torch.nn.utils.rnn.pad_sequence(tgt_inp, True, -1)
tgt_pmask = torch.nn.utils.rnn.pad_sequence(tgt_pmask, True, float("-inf"))

# Running it with and without no_grad(), and comparing:
output1 = tf(
    src=src_inp,
    tgt=tgt_inp,
    tgt_mask=tf.generate_square_subsequent_mask(tgt_inp.size(1)),
    src_key_padding_mask=src_pmask,
    tgt_key_padding_mask=tgt_pmask,
    memory_key_padding_mask=src_pmask
)

with torch.no_grad():
    output2 = tf(
        src=src_inp,
        tgt=tgt_inp,
        tgt_mask=tf.generate_square_subsequent_mask(tgt_inp.size(1)),
        src_key_padding_mask=src_pmask,
        tgt_key_padding_mask=tgt_pmask,
        memory_key_padding_mask=src_pmask
    )

print(torch.all(output1 == output2))