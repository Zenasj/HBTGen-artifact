import torch
import torch._dynamo as dynamo
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b
explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = (
    dynamo.explain(toy_example, torch.randn(10), torch.randn(10))
)
print(explanation_verbose)
"""
Dynamo produced 3 graphs, with 2 graph breaks and 6 ops.
 Break reasons:
1. call_function BuiltinVariable(print) [ConstantVariable(str)] {}
   File "t2.py", line 16, in toy_example
    print("woo")

2. generic_jump
   File "t2.py", line 17, in toy_example
    if b.sum() < 0:
 """