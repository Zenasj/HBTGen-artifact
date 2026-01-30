import torch
import torch.nn as nn


def transformer_encoder(inputs, input_seq_len):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        dropout=0.1,
        activation='relu',
        batch_first=True,
    )
    encoder_norm = nn.LayerNorm(16)
    encoder = nn.TransformerEncoder(
        encoder_layer, 2, encoder_norm
    )

    src_mask = torch.ones(inputs.shape[1], inputs.shape[1], dtype=torch.bool).triu_(diagonal=1)
    padding_mask = (torch.arange(inputs.shape[1])[None, :].cpu() >= input_seq_len[:, None])
    
    print(src_mask,src_mask.dtype, )
    print(padding_mask,padding_mask.dtype, )
    
    return encoder(inputs, 
        mask=src_mask,
        src_key_padding_mask=padding_mask,
    )
    
    

transformer_encoder_opt = torch.compile(transformer_encoder)

inputs = torch.randn(2,3,16)
input_seq_len = torch.tensor([3,2])

transformer_encoder_opt(inputs, input_seq_len)