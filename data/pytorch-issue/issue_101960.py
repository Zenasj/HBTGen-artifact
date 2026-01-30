import torch

def test_trace_out_operator_with_two_output():
    example_input = torch.rand(2, 8)
    out_1, out_2 = torch.cummax(example_input, 1)

    def run_cummax(example_input, out_1, out_2):
        output_1, output_2 = torch.cummax(example_input, 1, out=(out_1, out_2))
        return output_1, output_2

    trace_model = torch.jit.trace(run_cummax, (example_input, out_1, out_2))