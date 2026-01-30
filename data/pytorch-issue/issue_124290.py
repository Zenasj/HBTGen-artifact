# Whenever we allocate a fresh unbacked Symbol, we add it to this
        # pending list.  Unbacked symbol allocation can occur at unpredictable
        # points during meta tensor propagation, but at some point, the we
        # have to know what the binding site for an unbacked symbol is, and
        # this is computed when we actually place the node in the graph.  The
        # important thing is that we always actually handle every unaccounted
        # for unbacked symbol, so this list helps us keep track of them and
        # then make sure they are all accounted for.
        #
        # We could potentially give rise to errors earlier by lexically
        # scoping when we do propagation, and only allowing unbacked symbols
        # to be allocated at this point in time.  However this is inconvenient
        # to do in Dynamo, because fake tensor propagation is far from when we
        # analyze binding sites (set_example_value), so we do it in a more
        # mutatey way.
        #
        # NB: fresh unbacked symbols NEVER get substitutions applied to them,
        # they are binding sites!

import torch

@torch.compile(fullgraph=True)
def f(x, y):
    return torch.ops.aten.dropout.default(x, y, True)

f(torch.randn(10), torch.tensor(0.5))