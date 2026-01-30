if transposed:
            kernel = dilation_ * (weight[d] - 1)
            output_size.append((input[d] - 1) * stride[d - 2] - 2 * padding[d - 2] + kernel + 1)