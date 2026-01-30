import torch
import torch.nn as nn


def transformer_decoder(inputs, input_seq_len, memory):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        dropout=0.1,
        activation='relu',
        batch_first=True,
    )
    decoder_norm = nn.LayerNorm(16)
    decoder = nn.TransformerDecoder(
        decoder_layer, 2, decoder_norm
    )

    src_mask = torch.ones(inputs.shape[1], inputs.shape[1], dtype=torch.bool).triu_(diagonal=1)
    padding_mask = (torch.arange(inputs.shape[1])[None, :].cpu() >= input_seq_len[:, None])
    
    return decoder(inputs, 
                   memory,
        tgt_mask=src_mask,
       tgt_key_padding_mask=padding_mask,
        memory_key_padding_mask=padding_mask,
    )
    
inputs = torch.randn(2,3,16)
memory = torch.randn(2,3,16)
input_seq_len = torch.tensor([3,2])

transformer_decoder(inputs, input_seq_len, memory)