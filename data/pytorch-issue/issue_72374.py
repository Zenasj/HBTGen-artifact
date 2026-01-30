import torch.fx.experimental.diagnostics as diag

with diag.collect_when_fail():
    ...
    # in places where you want to dump diagnostics:
    diag.write("module.graph", str(module.graph))
    diag.write("some_bytes", pickle.dumps(module.some_param))
    # also supports retrieving data from a lambda. If it throws, it'll not impact
    # business logic:
    diag.write("some_data_2", lambda: this_might_throw())
    ...

    some_code_that_throws()  # this will trigger diagnostics to be uploaded
    ...