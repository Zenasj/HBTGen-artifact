for d in range(len(kernel_size)):
    default_size.append((input_size[-len(kernel_size) + d] - 1) * stride[d] + kernel_size[d] - 2 * padding[d])