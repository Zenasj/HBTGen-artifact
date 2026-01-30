mask = generate_padding_mask(src.size()[1], lengths)

mask = ~generate_padding_mask(src.size()[1], lengths)