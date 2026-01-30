def runAndSaveRNG(self, func, inputs, kwargs=None, input_func=None):
    kwargs = kwargs if kwargs else {}
    with freeze_rng_state():
        if input_func is not None:
            inputs = [input_func(*inputs, **kwargs)]
        results = func(*inputs, **kwargs)
    return results