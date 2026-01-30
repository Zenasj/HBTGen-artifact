aten.foo(my_tensor_input, "my_default_value")

aten.foo(my_tensor_input)  # the schema matcher will insert new_arg="my_default_value"