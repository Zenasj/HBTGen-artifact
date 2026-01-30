import torch

def test_dynamic_shape_topk_cpu():
    print("Starting the test.................")
    sizes = [5, 10, 15, 18, 16]

    def raw_function(t):
        k = t.shape[0] // 5
        out = torch.topk(t, k)
        value0 = out[0]
        value1 = out[1]
        return value1

    compiled_function_training = torch.compile(raw_function, dynamic=True)

    for s in sizes:
        t = torch.randn(s)
        result_compile_train = compiled_function_training(t)
        print("Output:", result_compile_train)