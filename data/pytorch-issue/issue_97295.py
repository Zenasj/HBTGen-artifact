import torch

def my_custom_function(x):
    return x + 1

def test_allow_in_graph():
    torch._dynamo.allow_in_graph(my_custom_function)
    @torch._dynamo.optimize("aot_eager")
    def fn(a):
        x = torch.add(a, 1)
        x = torch.add(x, 1)
        x = my_custom_function(x)
        x = torch.add(x, 1)
        x = torch.add(x, 1)
        return x

    fn(torch.randn(10))