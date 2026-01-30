def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, **kwargs):

    # Delayed operation here
    make_input = delayed(partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad))
    return [ModuleInput(constructor_input=FunctionInput(10, 8),
                        forward_input=FunctionInput(make_input((4, 10))),
                        desc="default")]

class TestModule(TestCase):
    @modules(module_db)
    def test_forward(self, device, dtype, module_info, module_input):

        # Call compute to construct tensors
        construct_args, construct_kwargs = module_input.constructor_input.compute()
        ...