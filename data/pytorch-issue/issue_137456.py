import torch

def test():
    # Defining `get_inner` in a local scope, outside @torch.compile-ed function,
    # so that it'll trigger the optimization logic to turn a read-only cell
    # variable into a variable of the cell content.
    x = torch.ones(1)
    def get_inner():
        def inner():
            return x + x

        # Calling `inner` so Dynamo won't skip this frame.
        inner()

        # Returning `inner` so the underlying `VariableTracker` representation
        # of `x` will escape to the root frame of tracing, where we force its
        # reconstruction.
        return inner

    @torch.compile
    def func():
        return get_inner()
    func()

test()