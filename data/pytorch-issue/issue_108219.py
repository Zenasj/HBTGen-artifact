if groups <= 0:
    raise ValueError('groups must be a positive integer')
if in_channels % groups != 0:
    raise ValueError('in_channels must be divisible by groups')
if out_channels % groups != 0:
    raise ValueError('out_channels must be divisible by groups')