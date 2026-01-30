is_magic_method = name[:2] == '__' and name[-2] == '__'
is_inpalce = name[-1] == "_" and not is_magic_method
self_variable = create_input((self_size,), dtype=dtype, device=device)[0][0]
# FixMe: run grad checks on inplace self
if is_inplace:
    self_variable.requires_grad = False