import torch
import torch.nn as nn

def test_graph_rng_module():
    # nn.dropout2d/nn.dropout both fails
    module = nn.Dropout2d(p=0.2)
    input = torch.randn(20, 16, 32, 32).to(device="cuda").requires_grad_()

    module_g = nn.Dropout2d(p=0.2)
    module_g = torch.cuda.make_graphed_callables(module_g, (input,))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    eager_output = input.clone()
    for _ in range(3):
        eager_out2 = module(eager_output)
        assert not torch.allclose(eager_out2, eager_output)
        eager_output = eager_out2


    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    graph_out = input.clone()
    for i in range(3):
        graph_out2 = module_g(graph_out)
        # fails on this assert because graph_out2 == graph_out
        assert not torch.allclose(graph_out2, graph_out), 'module_g forget to update rng state'
        graph_out = graph_out2

    assert torch.allclose(eager_output, graph_out)

graph_out = graph_out2

for i in range(3):
        graph_out2 = module_g(graph_out)
        print(graph_out.abs().sum())
        graph_out = graph_out2
        print(graph_out.abs().sum())

def test_graph_rng_functional():
    # copied from test_cuda, should work
    ops_with_kwargs = (
        (torch.nn.functional.dropout, {"p": 0.1}),
        (torch.nn.functional.rrelu, {"training": True}),
    )
    size = 10000

    def run(op, kwargs):
        a = torch.randn((size,), device="cuda", dtype=torch.float)

        # Control
        torch.cuda.manual_seed(5)
        eager_out = a
        for _ in range(6):
            eager_out = op(eager_out, **kwargs)

        graph_in = a.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda.manual_seed(5)

            g = torch.cuda.CUDAGraph()
            torch.cuda.empty_cache()
            g.capture_begin()
            graph_out = graph_in
            for _ in range(2):
                graph_out = op(graph_out, **kwargs)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(stream)

        # Runs a graphed->eager->graphed sequence of RNG ops.
        # replay() plays 2 invocations of the op, so the sequence has 6
        # invocations total, matching Control.
        # replay() reads from graph_in and writes to graph_out.
        g.replay()
        out = op(graph_out, **kwargs)
        out = op(out, **kwargs)
        graph_in.copy_(out)
        g.replay()

        # If replay() updated RNG state correctly, graph_out
        # should now hold data equal to eager_out.
        try:
            assert torch.allclose(eager_out, graph_out)
        except Exception as e:
            raise RuntimeError("Failed on ", op) from e

        # a different seed should make the result mismatch
        torch.cuda.manual_seed(10)
        graph_in.copy_(a)
        for _ in range(3):
            g.replay()
            graph_in.copy_(graph_out)

        try:
            assert not torch.allclose(eager_out, graph_out)
        except Exception as e:
            raise RuntimeError("Failed on ", op) from e

        # We hold references to all tensors used across streams up til this sync,
        # so no need to call record_stream on those tensors.
        torch.cuda.synchronize()

    for op, kwargs in ops_with_kwargs:
        run(op, kwargs)