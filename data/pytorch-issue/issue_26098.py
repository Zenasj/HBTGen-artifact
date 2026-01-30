if key_padding_mask is not None:
   attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
   attn_output_weights = attn_output_weights.masked_fill(
        key_padding_mask.unsqueeze(1).unsqueeze(2),
        float('-inf'),
   )
   attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)