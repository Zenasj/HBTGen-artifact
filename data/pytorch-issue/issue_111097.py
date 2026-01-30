import onnx

def compute_tensor_size(tensor):
    # Compute the size of the tensor based on its shape and data type
    size = tensor.size * tensor.itemsize
    return size

def sum_constant_and_initializer_sizes(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    total_size = 0
    initializer_size = 0
    constant_size = 0

    # Compute the size of constant nodes
    for node in model.graph.node:
        if node.op_type == 'Constant':
            constant_value = node.attribute[0].t
            # Convert constant value to numpy array
            constant_array = onnx.numpy_helper.to_array(constant_value)
            # Compute the size of the constant tensor
            tensor_size = compute_tensor_size(constant_array)
            total_size += tensor_size
            constant_size += tensor_size

    # Compute the size of initializer nodes that are not graph inputs
    for initializer in model.graph.initializer:
        if initializer.name not in [input.name for input in model.graph.input]:
            # Convert the shape and data type information to calculate size
            # tensor = onnx.helper.tensor_value_info_to_tensor(input)
            tensor = onnx.numpy_helper.to_array(initializer)
            tensor_size = compute_tensor_size(tensor)
            total_size += tensor_size
            initializer_size += tensor_size

    return total_size, constant_size, initializer_size

model_path = '/path/to/model.onnx'
total_size, constant_size, initializer_size = sum_constant_and_initializer_sizes(model_path)

print("Total size of constant nodes in bytes:", constant_size)
print("Total size of initializer nodes (excluding graph inputs) in bytes:", initializer_size)
print("Total size of constant and initializer nodes (excluding graph inputs) in bytes:", total_size)