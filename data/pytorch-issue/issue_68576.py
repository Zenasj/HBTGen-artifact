if num_workers > 0:
    if prefetch_factor is None:
        prefetch_factor = 2   # default value
else:
    if prefetch_factor is not None:
        raise ValueError('prefetch_factor option could only be specified in multiprocessing.' 
                         'let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.')