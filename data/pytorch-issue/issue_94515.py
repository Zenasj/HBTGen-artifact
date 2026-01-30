position_ids = attention_mask.long().cumsum(-1) - 1

position_ids = attention_mask.int().cumsum(-1) - 1