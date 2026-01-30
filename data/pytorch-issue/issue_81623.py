import torch
import torch.nn as nn

# fetch a data from dataloader, and the data is a dictionary
# and the example_inputs_dict is like: {key1:value1, key3:value3, key2:value2}
# the forward() is like: def forward(self, key1=value1, key2=value2, key3=value3)
example_inputs_dict = next(iter(dataloader))
jit_model = model.eval()
# use the dictionary to trace the model
jit_model = torch.jit.trace(jit_model, example_inputs_dict, strict=False)  # Now the IR will be graph(%self : __torch__.module.___torch_mangle_n.Mymodule, %key1 : type1, %key3 : type3, %key2 : type2)
jit_model = torch.jit.freeze(jit_model)

# It's OK to use dict as the parameter for traced model
jit_model(**example_inputs_dict)

example_inputs_tuple = (value1, value3, value2)
# It's wrong to rely on the original args order.
jit_model(*example_inputs_tuple)

def test_dictionary_as_example_inputs_for_jit_trace(self):
        class TestModule_v1(torch.nn.Module):
            def __init__(self):
                super(TestModule_v1, self).__init__()

            def forward(self, key2=None, key3=None, key4=None, key5=None, key1=None, key6=None):
                return key1 + key2 + key3

        class TestModule_v2(torch.nn.Module):
            def __init__(self):
                super(TestModule_v2, self).__init__()

            def forward(self, x, y):
                return x + y

        model_1 = TestModule_v1()
        model_2 = TestModule_v2()
        value1 = torch.ones(1)
        value2 = torch.ones(1)
        value3 = torch.ones(1)
        example_input_dict = {'key1': value1, 'key2': value2, 'key3': value3}
        traced_model_1 = torch.jit.trace(model_1, example_input_dict, strict=False)
        traced_model_2 = torch.jit.trace(model_2, {'x': torch.rand([2]), 'y': torch.rand([2])})
        res_1 = traced_model_1(**example_input_dict)
        self.assertEqual(res_1, 3 * torch.ones(1)) # Positive
        with self.assertRaisesRegex(RuntimeError, "forward\(\) is missing value for argument 'x'."):
            res_2 = traced_model_2(**{'z': torch.rand([2]), 'y': torch.rand([2])}) # Negative
        with self.assertRaisesRegex(RuntimeError, "forward\(\) is missing value for argument 'y'."):
            res_2 = traced_model_2(**{'x': torch.rand([2]), 'z': torch.rand([2])}) # Negative