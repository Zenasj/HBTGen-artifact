py
# case 1
make_fx(foo)(a, b) # raises RuntimeError: Tracing expected 3 arguments but got 2 concrete arguments

# case 2
make_fx(foo)(a, b, alpha=alpha) # TypeError: wrapped() got an unexpected keyword argument 'alpha'

# case 3
make_fx(foo)(a, b, alpha)(a, b, alpha=alpha) # TypeError: forward() got an unexpected keyword argument 'alpha'

# case 4
make_fx(foo)(a, b, alpha)(a, b) # works