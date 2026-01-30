n = a.numel() 
a.resize_(new_shape)
assert a.is_contiguous()
b = a.flatten()  # Does not copy
b[n:].zero_()